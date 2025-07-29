def cpq_search(query: str, search_type: str = "product", max_results: int = 5) -> str:
    """Searches for CPQ (Configure, Price, Quote) related information including products, configurations, and pricing.
    
    Args:
        query (str): The search query for products, configurations, or pricing
        search_type (str): Type of search - "product", "configuration", "pricing", or "compatibility"
        max_results (int): Maximum number of results to return (default: 5)
        
    Returns:
        Formatted string with bullet points of search results including product details, specifications, and pricing
        
    Examples:
        {"query": "laptop configurations", "search_type": "product", "max_results": 3}
        {"query": "Intel i7 processor compatibility", "search_type": "compatibility", "max_results": 5}
        {"query": "Dell Latitude pricing", "search_type": "pricing", "max_results": 3}
    """
    
    # Mock CPQ database - in a real implementation, this would connect to a product database
    product_database = {
        "laptop": [
            {
                "name": "Dell Latitude 5520",
                "cpu": "Intel i7-1165G7",
                "ram": "16GB DDR4",
                "storage": "512GB SSD",
                "price": "$1,299",
                "features": ["Backlit keyboard", "Fingerprint reader", "HDMI port"]
            },
            {
                "name": "HP EliteBook 840 G8",
                "cpu": "Intel i5-1135G7", 
                "ram": "8GB DDR4",
                "storage": "256GB SSD",
                "price": "$899",
                "features": ["Privacy screen", "Docking station compatible", "USB-C"]
            },
            {
                "name": "Lenovo ThinkPad X1 Carbon",
                "cpu": "Intel i7-1165G7",
                "ram": "32GB DDR4", 
                "storage": "1TB SSD",
                "price": "$1,599",
                "features": ["Carbon fiber chassis", "14-inch 4K display", "Thunderbolt 4"]
            },
            {
                "name": "Apple MacBook Pro 13",
                "cpu": "Apple M1",
                "ram": "16GB Unified Memory",
                "storage": "512GB SSD", 
                "price": "$1,499",
                "features": ["Retina display", "Touch Bar", "Up to 20 hours battery"]
            },
            {
                "name": "ASUS ZenBook 14",
                "cpu": "AMD Ryzen 7 5800H",
                "ram": "16GB DDR4",
                "storage": "512GB SSD",
                "price": "$1,199", 
                "features": ["NumberPad", "ErgoLift hinge", "Military-grade durability"]
            }
        ],
        "desktop": [
            {
                "name": "Dell OptiPlex 7090",
                "cpu": "Intel i7-11700",
                "ram": "16GB DDR4",
                "storage": "512GB SSD",
                "price": "$899",
                "features": ["Small form factor", "VESA mountable", "Energy Star certified"]
            },
            {
                "name": "HP EliteDesk 800 G6",
                "cpu": "Intel i5-10500",
                "ram": "8GB DDR4", 
                "storage": "256GB SSD",
                "price": "$699",
                "features": ["Tool-less design", "Multiple display support", "Security lock slot"]
            }
        ],
        "server": [
            {
                "name": "Dell PowerEdge R750",
                "cpu": "Intel Xeon Silver 4314",
                "ram": "64GB DDR4",
                "storage": "2TB SSD",
                "price": "$3,999",
                "features": ["2U rack mount", "Redundant power supplies", "iDRAC management"]
            }
        ]
    }
    
    compatibility_rules = {
        "Intel i7": {
            "compatible_ram": ["8GB", "16GB", "32GB", "64GB"],
            "compatible_storage": ["256GB", "512GB", "1TB", "2TB"],
            "min_power": "65W",
            "socket_type": "LGA1200"
        },
        "Intel i5": {
            "compatible_ram": ["4GB", "8GB", "16GB", "32GB"], 
            "compatible_storage": ["128GB", "256GB", "512GB", "1TB"],
            "min_power": "65W",
            "socket_type": "LGA1200"
        },
        "AMD Ryzen 7": {
            "compatible_ram": ["8GB", "16GB", "32GB", "64GB"],
            "compatible_storage": ["256GB", "512GB", "1TB", "2TB"], 
            "min_power": "65W",
            "socket_type": "AM4"
        }
    }
    
    try:
        query_lower = query.lower()
        results = []
        
        if search_type == "product":
            # Search for products matching the query
            for category, products in product_database.items():
                for product in products:
                    if any(term in product["name"].lower() or term in str(product).lower() 
                           for term in query_lower.split()):
                        results.append(product)
                        
        elif search_type == "configuration":
            # Search for specific configurations
            for category, products in product_database.items():
                for product in products:
                    if any(term in str(product).lower() for term in query_lower.split()):
                        results.append(product)
                        
        elif search_type == "pricing":
            # Search for pricing information
            for category, products in product_database.items():
                for product in products:
                    if any(term in product["name"].lower() or term in str(product).lower() 
                           for term in query_lower.split()):
                        results.append(product)
                        
        elif search_type == "compatibility":
            # Search for compatibility information
            for component, rules in compatibility_rules.items():
                if component.lower() in query_lower:
                    results.append({
                        "component": component,
                        "compatibility": rules
                    })
        
        # Limit results
        results = results[:max_results]
        
        if not results:
            return f"No {search_type} results found for query: '{query}'"
        
        # Format results
        formatted_results = []
        for result in results:
            if "compatibility" in result:
                # Compatibility result
                comp = result["compatibility"]
                formatted_results.append(
                    f"• {result['component']} Compatibility:\n"
                    f"  Compatible RAM: {', '.join(comp['compatible_ram'])}\n"
                    f"  Compatible Storage: {', '.join(comp['compatible_storage'])}\n"
                    f"  Minimum Power: {comp['min_power']}\n"
                    f"  Socket Type: {comp['socket_type']}"
                )
            else:
                # Product result
                features_str = ", ".join(result["features"])
                formatted_results.append(
                    f"• {result['name']} - {result['cpu']}, {result['ram']}, {result['storage']} - {result['price']}\n"
                    f"  Features: {features_str}"
                )
        
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching CPQ database: {str(e)}" 