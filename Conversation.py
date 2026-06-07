import json
from LanguageModel import LanguageModel
import os
from experiment_utils import (
    extract_price_from_text,
    looks_like_no_price,
    parse_price,
    price_candidates_from_text,
)

class Conversation:
    def __init__(self, product_data, buyer_model="gpt-3.5-turbo", seller_model="gpt-3.5-turbo", summary_model="gpt-3.5-turbo", max_turns=20, experiment_num=0, budget=None):
        self.product_data = product_data
        self.buyer_model_name = buyer_model  # Store the model name
        self.seller_model_name = seller_model  # Store the model name
        self.summary_model_name = summary_model  # Store the summary model name
        self.buyer_model = LanguageModel(model_name=buyer_model)
        self.seller_model = LanguageModel(model_name=seller_model)
        self.summary_model = LanguageModel(model_name=summary_model)
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
        # Add the original retail price as the first element
        retail_price_str = product_data["Retail Price"]
        self.seller_price_offers.append(parse_price(retail_price_str))
        # Most recent seller price offer
        self.current_price_offer = None
        # Flag to indicate if negotiation was completed successfully
        self.negotiation_completed = False
        # Result of the negotiation (accepted, rejected, or None if not completed)
        self.negotiation_result = None
        
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
        
        extracted_response = self.summary_model.get_response(prompt).strip()
        
        event = {
            "seller_message": seller_message,
            "summary_response": extracted_response,
            "source": None,
            "price": None,
            "status": None,
        }

        if looks_like_no_price(extracted_response):
            event["status"] = "no_price"
            self.price_extraction_events.append(event)
            return None
        
        print(f"Raw extracted price: '{extracted_response}'")

        price_value = extract_price_from_text(extracted_response, allow_bare_number=True)
        if price_value is not None:
            event.update({"source": "summary_response", "price": price_value, "status": "parsed"})
            self.price_extraction_events.append(event)
            print(f"Successfully extracted price: {price_value}")
            return price_value

        # Fallback only when the seller message has exactly one currency-marked price.
        # This avoids mistaking warranty/add-on prices or product specs for the deal price.
        seller_candidates = price_candidates_from_text(seller_message, allow_bare_number=False)
        if len(seller_candidates) == 1:
            price_value = seller_candidates[0]
            event.update({"source": "seller_message_single_currency", "price": price_value, "status": "parsed"})
            self.price_extraction_events.append(event)
            print(f"Extracted fallback seller-message price: {price_value}")
            return price_value

        event["status"] = "unparsed"
        event["seller_currency_candidates"] = seller_candidates
        self.price_extraction_events.append(event)
        print(f"Warning: Could not parse a unique price from '{extracted_response}'")
        return None
        
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
        evaluation = self.summary_model.get_response(evaluation_prompt).strip()
        
        # Process the evaluation result
        if "ACCEPTANCE" in evaluation:
            self.negotiation_completed = True
            self.negotiation_result = "accepted"
            return True
        elif "REJECTION" in evaluation:
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
        
        buyer_intro = self.buyer_model.get_response(intro_prompt)
        self.conversation_history.append({"speaker": "Buyer", "message": buyer_intro})
        print(f"\n[Initial] Buyer: {buyer_intro}")
        
        # Initialize current_price_offer with retail price
        self.current_price_offer = self.seller_price_offers[0]
        
        turn_count = 1
        # Continue the conversation until a stopping condition is met or safety max_turns reached
        while turn_count <= self.max_turns:
            # Seller's turn
            seller_messages = self.format_seller_prompt()
            seller_response = self.seller_model.get_chat_response(seller_messages)
            self.conversation_history.append({"speaker": "Seller", "message": seller_response})
            print(f"\n[Turn {turn_count}] Seller: {seller_response}")
            
            # Extract price offer using summary_model
            price_offer = self.extract_price_from_seller_message(seller_response)
            
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
            buyer_response = self.buyer_model.get_chat_response(buyer_messages)
            self.conversation_history.append({"speaker": "Buyer", "message": buyer_response})
            print(f"\n[Turn {turn_count}] Buyer: {buyer_response}")
            
            # Check if negotiation should end
            if self.evaluate_negotiation_state():
                print(f"\nNegotiation has concluded with result: {self.negotiation_result}")
                break
                
            turn_count += 1
        
        # Record the actual number of turns completed
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
            "budget": self.budget,  # Include budget in the saved data
            "budget_scenario": self.budget_scenario,  # Include budget scenario name
            "completed_turns": self.completed_turns,  # Include the actual number of turns
            "negotiation_completed": self.negotiation_completed,  # Whether negotiation reached conclusion
            "negotiation_result": self.negotiation_result,  # Result of the negotiation
            "models": {
                "buyer": self.buyer_model_name,
                "seller": self.seller_model_name,
                "summary": self.summary_model_name
            },
            "parameters": {
                "max_turns": self.max_turns
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Conversation saved to: {output_file}")
