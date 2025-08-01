def search_product(query: str) -> str:
    """Searches for products and returns a suitable product configuration.
    
    Args:
        query (str): The search query for products, configurations, or pricing
        
    Returns:
        Product configuration in JSON format
        
    Examples:
        {"query": "laptop configurations"}
        {"query": "Intel i7 processor compatibility"}
        {"query": "Dell Latitude pricing"}
    """
    
    product = {
        "Product": "Dell Latitude 5520",
        "Features": ["Intel i7-1165G7", "16GB DDR4", "512GB SSD", "Backlit keyboard", "Fingerprint reader"],
        "Price": "$1,299"
    }
    
    return str(product) 