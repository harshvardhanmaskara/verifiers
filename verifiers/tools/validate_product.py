def validate_product(product: dict) -> str:
    """Validates product configuration format.
    
    Args:
        product (dict): Product configuration to validate (e.g., {"Product": "name", "Features": ["list"], "Price": "price"})
        
    Returns:
        "Valid" or "Not Valid" based on JSON format validation
        
    Examples:
        {"product": {"Product": "Dell Latitude", "Features": ["Intel i7", "16GB"], "Price": "$1,299"}}
    """
    
    try:
        # Check if product has the required fields
        required_fields = ["Product", "Features", "Price"]
        
        for field in required_fields:
            if field not in product:
                return "Not Valid"
        
        # Check if Features is a list
        if not isinstance(product["Features"], list):
            return "Not Valid"
        
        # Check if Product and Price are strings
        if not isinstance(product["Product"], str) or not isinstance(product["Price"], str):
            return "Not Valid"
        
        return "Valid"
        
    except Exception as e:
        return "Not Valid" 