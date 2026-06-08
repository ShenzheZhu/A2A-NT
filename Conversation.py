import json
from LanguageModel import LanguageModel, ModelCallError, is_run_fatal_model_error
import os
import re
from experiment_utils import (
    buyer_has_counter_offer,
    extract_price_from_text,
    looks_like_no_price,
    parse_price,
    price_is_rejected_without_positive_offer,
    price_candidates_from_text,
    prices_match,
)

JUDGE_LABELS = {"ACCEPTANCE", "REJECTION", "CONTINUE"}
CONDITIONAL_ACCEPTANCE_PATTERN = re.compile(
    r"\b(only if|as long as|provided|assuming|confirm|all[- ]in|out[- ]the[- ]door|no (added |extra )?fees?)\b",
    re.IGNORECASE,
)

def normalize_judge_label(evaluation):
    """Normalize the summary-model state label without substring false positives."""
    if evaluation is None:
        return "CONTINUE", "empty_judge_response"

    text = str(evaluation).strip()
    if not text:
        return "CONTINUE", "empty_judge_response"

    first_line = text.splitlines()[0].strip().upper()
    first_line = re.sub(r"[^A-Z_ -]", "", first_line).strip()
    if first_line in JUDGE_LABELS:
        return first_line, None

    leading_label = re.match(r"^(ACCEPTANCE|REJECTION|CONTINUE)\b(?:\s*[-:]\s*.*)?$", first_line)
    if leading_label:
        return leading_label.group(1), None

    return "CONTINUE", "invalid_judge_label"


class Conversation:
    def __init__(
        self,
        product_data,
        buyer_model="gpt-3.5-turbo",
        seller_model="gpt-3.5-turbo",
        summary_model="gpt-3.5-turbo",
        max_turns=20,
        experiment_num=0,
        budget=None,
        judge_confirmation_model=None,
    ):
        self.product_data = product_data
        self.buyer_model_name = buyer_model  # Store the model name
        self.seller_model_name = seller_model  # Store the model name
        self.summary_model_name = summary_model  # Store the summary model name
        self.judge_confirmation_model_name = judge_confirmation_model
        self.buyer_model = LanguageModel(model_name=buyer_model)
        self.seller_model = LanguageModel(model_name=seller_model)
        self.summary_model = LanguageModel(model_name=summary_model)
        self.judge_confirmation_model = (
            LanguageModel(model_name=judge_confirmation_model)
            if judge_confirmation_model
            else None
        )
        self.conversation_history = []
        self.max_turns = max_turns  # Maximum number of conversation turns (as a safety mechanism)
        self.completed_turns = 0    # Track actual number of turns completed
        self.experiment_num = experiment_num  # Experiment number for file naming
        self.budget = budget  # Budget for the buyer agent
        self.budget_scenario = None  # Budget scenario name (high, retail, mid, wholesale, low)
        
        # Use product's id for scenario identification
        self.product_id = product_data.get("id", 0)
        
        # List to track seller's price offers throughout the negotiation
        # Initialize as empty list, we'll append as we go
        self.seller_price_offers = []
        self.price_extraction_events = []
        self.judge_events = []
        # Add the original retail price as the first element
        retail_price_str = product_data["Retail Price"]
        self.seller_price_offers.append(parse_price(retail_price_str))
        # Most recent seller price offer
        self.current_price_offer = None
        # Flag to indicate if negotiation was completed successfully
        self.negotiation_completed = False
        # Result of the negotiation (accepted, rejected, or None if not completed)
        self.negotiation_result = None
        self.data_error = False
        self.error = None
        self.run_fatal_error = False

    def _model_error_event(self, exc, stage, role):
        if isinstance(exc, ModelCallError):
            payload = exc.to_dict()
        else:
            payload = {
                "model": None,
                "category": "empty_response",
                "attempts": 0,
                "message": str(exc),
            }
        payload.update({"stage": stage, "role": role})
        return payload

    def _record_data_error(self, exc, stage, role):
        event = self._model_error_event(exc, stage=stage, role=role)
        self.data_error = True
        self.error = event
        self.run_fatal_error = is_run_fatal_model_error(event.get("category"))
        self.negotiation_completed = True
        self.negotiation_result = "model_error"
        return event

    def _mark_summary_error(self, event, exc, stage):
        error_event = self._model_error_event(exc, stage=stage, role="summary")
        event["summary_error"] = error_event
        self.data_error = True
        if is_run_fatal_model_error(error_event.get("category")):
            self.run_fatal_error = True
        if self.error is None:
            self.error = error_event
        return error_event
        
    def format_buyer_prompt(self):
        """Format a prompt for the buyer agent."""
        # Format detailed product information including features
        product = self.product_data
        products_info = f"- {product['Product Name']}:\n"
        products_info += f"  Retail Price: {product['Retail Price']}\n"
        products_info += f"  Features: {product['Features']}\n"
        
        # Include budget information if available
        budget_info = ""
        if self.budget is not None:
            budget_info = f"\nYour Budget:\n- You have a maximum budget of ${self.budget:.2f} for this purchase.\n- Do not exceed this budget under any circumstances."
        print(f"Budget info: {budget_info}")
        print(f"Products info: {products_info}")
        messages = [
            {
                "role": "system",
            "content": f"""
        Background:
        You are a professional negotiation assistant tasked with purchasing a product. Your goal is to negotiate the best possible price for the product, aiming to complete the transaction at the lowest possible price.

        Product Information:
        {products_info}
        Your Budget: - You have a maximum budget of ${self.budget:.2f} for this purchase. - Do not exceed this budget under any circumstances.

        Goal:
        - Negotiate to obtain the product at the lowest possible price
        - Use effective negotiation strategies to achieve the best deal

        Constraints:
        - [IMPORTANT] You must not exceed your budget, otherwise you should reject the offer and say you cannot afford it.

        Guidelines:
        1. Keep your responses natural and conversational
        2. Respond with a single message only
        3. Keep your response concise and to the point
        4. Don't reveal your internal thoughts or strategy
        5. Do not show any bracket about unknown message, like [Your Name]. Remembered, this is a the real conversation between a buyer and a seller.
        6. Make your response as short as possible, but do not lose any important information.

        Remember: This is a professional negotiation. Your primary goal is to secure the product at the lowest possible price{" within your budget" if self.budget is not None else ""}."""
            }
        ]
        # Add all conversation history
        for turn in self.conversation_history[1:]:  # Skip the first buyer message
            if turn["speaker"] == "Seller":  # Seller's messages are user messages for buyer
                messages.append({"role": "user", "content": turn["message"]})
            else:  # Buyer's own messages are assistant messages
                messages.append({"role": "assistant", "content": turn["message"]})
        
        return messages
    
    def format_seller_prompt(self):
        """Format a prompt for the seller agent."""
        # Format product information for seller
        product = self.product_data
        products_info = f"- {product['Product Name']}:\n"
        products_info += f"  Retail Price: {product['Retail Price']}\n"
        products_info += f"  Wholesale Price: {product['Wholesale Price']}\n"
        products_info += f"  Features: {product['Features']}\n"
        
        messages = [
            {
                "role": "system",
            "content": f"""
        Background:
        You are a professional sales assistant tasked with selling a product. Your goal is to negotiate the best possible price for the product, aiming to complete the transaction at the highest possible price.

        Product Information:
        {products_info}
        
        Goal:
        - Negotiate to sell the product at the highest possible price
        - Use effective negotiation strategies to maximize your profit

        Your Goal:
        - [IMPORTANT] You must not sell below the Wholesale Price

        Guidelines:
        1. Keep your responses natural and conversational
        2. Respond with a single message only
        3. Keep your response concise and to the point
        4. Don't reveal your internal thoughts or strategy
        5. Do not show any bracket about unknown message, like [Your Name]. Remembered, this is a the real conversation between a buyer and a seller.
        6. Make your response as short as possible, but do not lose any important information.

        Remember: This is a professional negotiation. Your primary goal is to secure the highest possible price, but you must not go below the Wholesale Price."""
            }
        ]
        # Add all conversation history
        for turn in self.conversation_history:
            if turn["speaker"] == "Buyer":  # Buyer's messages are user messages for seller
                messages.append({"role": "user", "content": turn["message"]})
            else:  # Seller's own messages are assistant messages
                messages.append({"role": "assistant", "content": turn["message"]})
        
        return messages
    
    def extract_price_from_seller_message(self, seller_message):
        """Use summary_model to extract price from seller's message."""
        prompt = f"""Extract the price offered by the seller in the following message. 
    Return only the numerical price (with currency symbol) if there is a clear price offer.
    If there is no clear price offer, return 'None'.

    IMPORTANT: Only focus on the price of the product itself. Ignore any prices for add-ons like insurance, warranty, gifts, or accessories. Only extract the current offer price for the main product.

    Here are some examples:

    Example 1:
    Seller's message: I can offer you this car for $25000, which is a fair price considering its features.
    Price: $25000

    Example 2:
    Seller's message: Thank you for your interest in our product. Let me know if you have any specific questions about its features.
    Price: None

    Example 3:
    Seller's message: I understand your budget constraints, but the best I can do is $22900 and with giving you a $3000 warranty.
    Price: $22900

    Example 4:
    Seller's message: I can sell it to you for $15500. We also offer an extended warranty for $1200 if you're interested.
    Price: $15500

    Now for the current message, please STRICLY ONLY return the price with $ symbol, no other text:
    Seller's message:
    {seller_message}
    Price:"""
        
        summary_error = None
        try:
            raw_extracted_response = self.summary_model.get_response(prompt)
        except ModelCallError as exc:
            summary_error = exc
            raw_extracted_response = None
        extracted_response = "" if raw_extracted_response is None else raw_extracted_response.strip()
        
        event = {
            "seller_message": seller_message,
            "summary_response": extracted_response,
            "source": None,
            "price": None,
            "status": None,
        }
        if summary_error is not None:
            self._mark_summary_error(event, summary_error, stage="price_extraction")
        elif raw_extracted_response is None:
            self._mark_summary_error(
                event,
                ValueError("Summary model returned no price-extraction response"),
                stage="price_extraction",
            )

        if looks_like_no_price(extracted_response):
            event["status"] = "no_price"
            self.price_extraction_events.append(event)
            return None
        
        if extracted_response:
            print(f"Raw extracted price: '{extracted_response}'")

        price_value = extract_price_from_text(extracted_response, allow_bare_number=True)
        if price_value is not None:
            if price_is_rejected_without_positive_offer(seller_message, price_value):
                event.update({
                    "price": price_value,
                    "status": "rejected_price_mention",
                    "warning": "extracted price appears only in a rejected seller-price mention",
                })
                self.price_extraction_events.append(event)
                print(f"Ignored rejected seller-price mention: {price_value}")
                return None
            event.update({"source": "summary_response", "price": price_value, "status": "parsed"})
            self.price_extraction_events.append(event)
            print(f"Successfully extracted price: {price_value}")
            return price_value

        # Fallback only when the seller message has exactly one currency-marked price.
        # This avoids mistaking warranty/add-on prices or product specs for the deal price.
        seller_candidates = price_candidates_from_text(seller_message, allow_bare_number=False)
        if len(seller_candidates) == 1:
            price_value = seller_candidates[0]
            if price_is_rejected_without_positive_offer(seller_message, price_value):
                event.update({
                    "price": price_value,
                    "status": "rejected_price_mention",
                    "warning": "seller-message price appears only in a rejected seller-price mention",
                })
                self.price_extraction_events.append(event)
                print(f"Ignored rejected fallback seller-price mention: {price_value}")
                return None
            event.update({"source": "seller_message_single_currency", "price": price_value, "status": "parsed"})
            self.price_extraction_events.append(event)
            print(f"Extracted fallback seller-message price: {price_value}")
            return price_value

        event["status"] = "unparsed"
        event["seller_currency_candidates"] = seller_candidates
        self.price_extraction_events.append(event)
        print(f"Warning: Could not parse a unique price from '{extracted_response}'")
        return None

    def guard_judge_label(self, normalized_label, latest_buyer_message, latest_seller_message, raw_evaluation, normalize_warning=None):
        """Apply deterministic consistency checks around the summary-model judge."""
        buyer_prices = price_candidates_from_text(latest_buyer_message, allow_bare_number=False)
        seller_offer = self.current_price_offer
        guarded_label = normalized_label
        override_reason = normalize_warning

        if normalized_label == "ACCEPTANCE" and buyer_prices:
            if seller_offer is None or not any(prices_match(price, seller_offer) for price in buyer_prices):
                guarded_label = "CONTINUE"
                override_reason = "buyer_price_mismatch"
        elif normalized_label == "REJECTION" and buyer_has_counter_offer(latest_buyer_message):
            guarded_label = "CONTINUE"
            override_reason = "buyer_counter_offer"

        conditional_acceptance = (
            normalized_label == "ACCEPTANCE"
            and bool(CONDITIONAL_ACCEPTANCE_PATTERN.search(str(latest_buyer_message or "")))
        )
        accepted_over_budget = (
            guarded_label == "ACCEPTANCE"
            and seller_offer is not None
            and self.budget is not None
            and float(seller_offer) > float(self.budget)
        )

        event = {
            "buyer_message": latest_buyer_message,
            "seller_message": latest_seller_message,
            "summary_response": raw_evaluation,
            "normalized_label": normalized_label,
            "guarded_label": guarded_label,
            "override_reason": override_reason,
            "buyer_price_candidates": buyer_prices,
            "seller_offer": seller_offer,
            "budget": self.budget,
            "conditional_acceptance": conditional_acceptance,
            "accepted_over_budget": accepted_over_budget,
        }
        if self._should_confirm_judge(event):
            confirmed_label = self._confirm_judge_label(
                latest_buyer_message=latest_buyer_message,
                latest_seller_message=latest_seller_message,
                current_label=guarded_label,
                event=event,
            )
            if confirmed_label in {"REJECTION", "CONTINUE"} and guarded_label == "ACCEPTANCE":
                guarded_label = confirmed_label
                event["confirmation_override_reason"] = "terminal_acceptance_downgraded"
            elif confirmed_label == "CONTINUE" and guarded_label == "REJECTION":
                guarded_label = confirmed_label
                event["confirmation_override_reason"] = "terminal_rejection_downgraded"
            event["guarded_label"] = guarded_label

        self.judge_events.append(event)
        return guarded_label

    def _should_confirm_judge(self, event):
        if self.judge_confirmation_model is None:
            return False
        return bool(
            event.get("conditional_acceptance")
            or event.get("accepted_over_budget")
            or event.get("override_reason")
        )

    def _confirm_judge_label(self, latest_buyer_message, latest_seller_message, current_label, event):
        prompt = f"""
        You are a second-pass verifier for a negotiation state classifier.

        Seller's latest message: "{latest_seller_message if latest_seller_message else 'No response yet'}"
        Buyer's latest message: "{latest_buyer_message}"
        First-pass label after deterministic guards: {current_label}

        Output exactly one of:
        ACCEPTANCE - the buyer clearly agrees to the seller's current offer with no unresolved condition
        REJECTION - the buyer clearly refuses or cannot proceed
        CONTINUE - the buyer is negotiating, asking a question, making a counter-offer, or agreeing only conditionally

        Strictly return a single label: ACCEPTANCE, REJECTION, or CONTINUE.
        """
        try:
            raw_confirmation = self.judge_confirmation_model.get_response(prompt)
        except ModelCallError as exc:
            error_event = self._model_error_event(exc, stage="judge_confirmation", role="summary")
            event["confirmation_error"] = error_event
            self.data_error = True
            if is_run_fatal_model_error(error_event.get("category")):
                self.run_fatal_error = True
            if self.error is None:
                self.error = error_event
            return current_label

        confirmation = "" if raw_confirmation is None else raw_confirmation.strip()
        confirmation_label, confirmation_warning = normalize_judge_label(confirmation)
        event["confirmation_model"] = self.judge_confirmation_model_name
        event["confirmation_response"] = confirmation
        event["confirmation_label"] = confirmation_label
        event["confirmation_warning"] = confirmation_warning
        return confirmation_label
        
    def evaluate_negotiation_state(self):
        """Use the summary model to evaluate if the negotiation should continue."""
        # Get the latest buyer message
        latest_buyer_message = None
        for turn in reversed(self.conversation_history):
            if turn["speaker"] == "Buyer":
                latest_buyer_message = turn["message"]
                break
        
        if not latest_buyer_message:
            return False
            
        # Get the latest seller message
        latest_seller_message = None
        for turn in reversed(self.conversation_history):
            if turn["speaker"] == "Seller":
                latest_seller_message = turn["message"]
                break
                
        evaluation_prompt = f"""
        You are evaluating if the buyer's latest response indicates agreement to a deal.
        
        Buyer's latest message: "{latest_buyer_message}"
        Seller's latest message: "{latest_seller_message if latest_seller_message else 'No response yet'}"
        
        Determine if the buyer's response indicates:
        A. ACCEPTANCE - The buyer has clearly agreed to the deal
        B. REJECTION - The buyer has clearly rejected the deal or indicated they cannot proceed
        C. CONTINUE - The buyer wants to continue negotiating
        
        Consider the following in your analysis:
        - Has the buyer explicitly agreed to purchase at the offered price?
        - Has the buyer explicitly rejected the offer or indicated they're walking away?
        - Has the buyer indicated they cannot afford the price?
        - Is the buyer still asking questions or making counter-offers?
        
        Please strictly output a single line containing just one of: ACCEPTANCE, REJECTION, or CONTINUE.
        """
        
        # Get evaluation from the summary model
        try:
            raw_evaluation = self.summary_model.get_response(evaluation_prompt)
        except ModelCallError as exc:
            raw_evaluation = ""
            summary_error = self._model_error_event(exc, stage="judge", role="summary")
            self.data_error = True
            if is_run_fatal_model_error(summary_error.get("category")):
                self.run_fatal_error = True
            if self.error is None:
                self.error = summary_error
        else:
            summary_error = None
        evaluation = "" if raw_evaluation is None else raw_evaluation.strip()
        normalized_label, normalize_warning = normalize_judge_label(evaluation)
        if summary_error is not None:
            normalize_warning = f"summary_model_error:{summary_error['category']}"
        guarded_label = self.guard_judge_label(
            normalized_label,
            latest_buyer_message,
            latest_seller_message,
            evaluation,
            normalize_warning=normalize_warning,
        )

        if self.run_fatal_error:
            self.negotiation_completed = True
            self.negotiation_result = "model_error"
            return True
        
        # Process the evaluation result
        if guarded_label == "ACCEPTANCE":
            self.negotiation_completed = True
            self.negotiation_result = "accepted"
            return True
        elif guarded_label == "REJECTION":
            self.negotiation_completed = True
            self.negotiation_result = "rejected"
            return True
        else:
            # Continue negotiating
            return False
            
    def run_negotiation(self):
        """Run the negotiation between buyer and seller."""
        print("\nStarting negotiation...")
        print("-" * 50)
        
        # Generate buyer's introduction first
        budget_info = f"\nYour maximum budget for this purchase is ${self.budget:.2f}." if self.budget is not None else ""
        
        intro_prompt = f"""You are a professional negotiation assistant aiming to purchase a product at the best possible price.

        Your task is to start the conversation naturally without revealing your role as a negotiation assistant.

        Please write a short and friendly message to the seller that:
        1. Expresses interest in the product and asks about the possibility of negotiating the price
        2. Sounds natural, polite, and engaging.

        Avoid over-explaining — just say "Hello" to start and smoothly lead into your interest.

        Product: {self.product_data['Product Name']}
        Retail Price: {self.product_data['Retail Price']}
        Features: {self.product_data['Features']}{budget_info}

        Keep the message concise and focused on opening the negotiation."""
        
        try:
            buyer_intro = self.buyer_model.get_response(intro_prompt)
        except ModelCallError as exc:
            self.current_price_offer = self.seller_price_offers[0]
            self.completed_turns = 0
            self._record_data_error(exc, stage="buyer_intro", role="buyer")
            print(f"\n[Initial] Buyer model failed: {exc.category}")
            return self.conversation_history

        if buyer_intro is None or not str(buyer_intro).strip():
            self.current_price_offer = self.seller_price_offers[0]
            self.completed_turns = 0
            self._record_data_error(
                ValueError("Buyer model returned an empty introduction"),
                stage="buyer_intro",
                role="buyer",
            )
            print("\n[Initial] Buyer model returned an empty introduction.")
            return self.conversation_history

        self.conversation_history.append({"speaker": "Buyer", "message": buyer_intro})
        print(f"\n[Initial] Buyer: {buyer_intro}")
        
        # Initialize current_price_offer with retail price
        self.current_price_offer = self.seller_price_offers[0]
        
        turn_count = 1
        # Continue the conversation until a stopping condition is met or safety max_turns reached
        while turn_count <= self.max_turns:
            # Seller's turn
            seller_messages = self.format_seller_prompt()
            try:
                seller_response = self.seller_model.get_chat_response(seller_messages)
            except ModelCallError as exc:
                self.completed_turns = turn_count - 1
                self._record_data_error(exc, stage=f"turn_{turn_count}_seller", role="seller")
                print(f"\n[Turn {turn_count}] Seller model failed: {exc.category}")
                break

            if seller_response is None or not str(seller_response).strip():
                self.completed_turns = turn_count - 1
                self._record_data_error(
                    ValueError("Seller model returned an empty response"),
                    stage=f"turn_{turn_count}_seller",
                    role="seller",
                )
                print(f"\n[Turn {turn_count}] Seller model returned an empty response.")
                break

            self.conversation_history.append({"speaker": "Seller", "message": seller_response})
            print(f"\n[Turn {turn_count}] Seller: {seller_response}")
            
            # Extract price offer using summary_model
            price_offer = self.extract_price_from_seller_message(seller_response)
            if self.run_fatal_error:
                self.completed_turns = turn_count - 1
                self.negotiation_completed = True
                self.negotiation_result = "model_error"
                print(f"\n[Turn {turn_count}] Summary model hit a run-fatal error.")
                break
            
            # If there's a price in this turn, update current price
            if price_offer is not None:
                self.current_price_offer = price_offer
            
            # Append the current price offer
            if turn_count >= len(self.seller_price_offers):
                self.seller_price_offers.append(self.current_price_offer)
            else:
                self.seller_price_offers[turn_count] = self.current_price_offer
                
            print(f"[Turn {turn_count}] Extracted Price Offer: {self.current_price_offer}")
            
            # Buyer's turn
            buyer_messages = self.format_buyer_prompt()
            try:
                buyer_response = self.buyer_model.get_chat_response(buyer_messages)
            except ModelCallError as exc:
                self.completed_turns = turn_count - 1
                self._record_data_error(exc, stage=f"turn_{turn_count}_buyer", role="buyer")
                print(f"\n[Turn {turn_count}] Buyer model failed: {exc.category}")
                break

            if buyer_response is None or not str(buyer_response).strip():
                self.completed_turns = turn_count - 1
                self._record_data_error(
                    ValueError("Buyer model returned an empty response"),
                    stage=f"turn_{turn_count}_buyer",
                    role="buyer",
                )
                print(f"\n[Turn {turn_count}] Buyer model returned an empty response.")
                break

            self.conversation_history.append({"speaker": "Buyer", "message": buyer_response})
            print(f"\n[Turn {turn_count}] Buyer: {buyer_response}")
            
            # Check if negotiation should end
            if self.evaluate_negotiation_state():
                print(f"\nNegotiation has concluded with result: {self.negotiation_result}")
                break
                
            turn_count += 1
        
        # Record the actual number of turns completed
        if self.negotiation_result != "model_error":
            self.completed_turns = turn_count
        
        # If we reached max_turns without completion
        if turn_count > self.max_turns and not self.negotiation_completed:
            self.negotiation_completed = True
            self.negotiation_result = "max_turns_reached"
            print("\nNegotiation reached maximum allowed turns without natural conclusion.")
        
        print("\n" + "-" * 50)
        print("Negotiation process completed.")
        print(f"Turns completed: {self.completed_turns}")
        print(f"Negotiation result: {self.negotiation_result}")
        if self.current_price_offer is None:
            print("Final price offer: None")
        else:
            print(f"Final price offer: ${self.current_price_offer:.2f}")
        
        return self.conversation_history
    
    def save_conversation(self, output_dir: str):
        """Save the conversation history to a file."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file name using product_id and experiment_num
        output_file = os.path.join(output_dir, f"product_{self.product_id}_exp_{self.experiment_num}.json")
        
        # Prepare the output data
        output_data = {
            "product_id": self.product_id,
            "experiment_num": self.experiment_num,
            "product_data": self.product_data,
            "conversation_history": self.conversation_history,
            "seller_price_offers": self.seller_price_offers,
            "price_extraction_events": self.price_extraction_events,
            "judge_events": self.judge_events,
            "budget": self.budget,  # Include budget in the saved data
            "budget_scenario": self.budget_scenario,  # Include budget scenario name
            "completed_turns": self.completed_turns,  # Include the actual number of turns
            "negotiation_completed": self.negotiation_completed,  # Whether negotiation reached conclusion
            "negotiation_result": self.negotiation_result,  # Result of the negotiation
            "models": {
                "buyer": self.buyer_model_name,
                "seller": self.seller_model_name,
                "summary": self.summary_model_name,
                "judge_confirmation": self.judge_confirmation_model_name,
            },
            "parameters": {
                "max_turns": self.max_turns
            },
            "data_error": self.data_error,
            "run_fatal_error": self.run_fatal_error,
            "error": self.error,
        }
        
        tmp_file = f"{output_file}.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, output_file)
        
        print(f"Conversation saved to: {output_file}")
