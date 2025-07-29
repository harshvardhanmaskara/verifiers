def cpq_validate(validation_type: str, data: dict, compatibility_check: bool = True) -> str:
    """Validates CPQ (Configure, Price, Quote) configurations, pricing, and compatibility.
    
    Args:
        validation_type (str): Type of validation - "configuration", "pricing", "compatibility", or "requirements"
        data (dict): Configuration data to validate (e.g., {"cpu": "Intel i7", "ram": "16GB", "storage": "512GB"})
        compatibility_check (bool): Whether to perform compatibility validation (default: True)
        
    Returns:
        Formatted string with validation results including status, issues, and recommendations
        
    Examples:
        {"validation_type": "configuration", "data": {"cpu": "Intel i7", "ram": "16GB", "storage": "512GB"}}
        {"validation_type": "pricing", "data": {"product": "Dell Latitude", "price": "$1,299"}}
        {"validation_type": "compatibility", "data": {"cpu": "Intel i7", "ram": "32GB"}}
    """
    
    # Mock validation rules and constraints
    validation_rules = {
        "configuration": {
            "required_fields": ["cpu", "ram", "storage"],
            "valid_cpus": ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9", "Apple M1", "Apple M2"],
            "valid_ram": ["4GB", "8GB", "16GB", "32GB", "64GB"],
            "valid_storage": ["128GB", "256GB", "512GB", "1TB", "2TB", "4TB"]
        },
        "compatibility": {
            "Intel i7": {
                "min_ram": "8GB",
                "max_ram": "64GB",
                "compatible_storage": ["256GB", "512GB", "1TB", "2TB"],
                "power_requirement": "65W"
            },
            "Intel i5": {
                "min_ram": "4GB", 
                "max_ram": "32GB",
                "compatible_storage": ["128GB", "256GB", "512GB", "1TB"],
                "power_requirement": "65W"
            },
            "AMD Ryzen 7": {
                "min_ram": "8GB",
                "max_ram": "64GB", 
                "compatible_storage": ["256GB", "512GB", "1TB", "2TB"],
                "power_requirement": "65W"
            },
            "Apple M1": {
                "min_ram": "8GB",
                "max_ram": "16GB",
                "compatible_storage": ["256GB", "512GB", "1TB"],
                "power_requirement": "30W"
            }
        },
        "pricing": {
            "budget_ranges": {
                "low": {"min": 0, "max": 800},
                "medium": {"min": 800, "max": 1500}, 
                "high": {"min": 1500, "max": 3000},
                "enterprise": {"min": 3000, "max": 10000}
            },
            "price_validation": {
                "Dell Latitude": {"min": 800, "max": 2000},
                "HP EliteBook": {"min": 600, "max": 1800},
                "Lenovo ThinkPad": {"min": 1000, "max": 2500},
                "Apple MacBook": {"min": 1200, "max": 3000}
            }
        }
    }
    
    try:
        results = []
        issues = []
        recommendations = []
        
        if validation_type == "configuration":
            # Validate configuration completeness and validity
            config = data
            
            # Check required fields
            for field in validation_rules["configuration"]["required_fields"]:
                if field not in config:
                    issues.append(f"Missing required field: {field}")
                elif config[field] not in validation_rules["configuration"][f"valid_{field}s"]:
                    issues.append(f"Invalid {field}: {config[field]}")
            
            # Check for valid configuration
            if not issues:
                results.append("✓ Configuration is valid")
                results.append("✓ All required fields are present")
                results.append("✓ All component specifications are valid")
                
                # Add compatibility check if requested
                if compatibility_check and "cpu" in config:
                    cpu = config["cpu"]
                    if cpu in validation_rules["compatibility"]:
                        comp_rules = validation_rules["compatibility"][cpu]
                        
                        if "ram" in config:
                            ram = config["ram"]
                            if ram < comp_rules["min_ram"]:
                                issues.append(f"RAM {ram} is below minimum {comp_rules['min_ram']} for {cpu}")
                            elif ram > comp_rules["max_ram"]:
                                issues.append(f"RAM {ram} exceeds maximum {comp_rules['max_ram']} for {cpu}")
                            else:
                                results.append(f"✓ RAM {ram} is compatible with {cpu}")
                        
                        if "storage" in config:
                            storage = config["storage"]
                            if storage not in comp_rules["compatible_storage"]:
                                issues.append(f"Storage {storage} may not be optimal for {cpu}")
                            else:
                                results.append(f"✓ Storage {storage} is compatible with {cpu}")
                        
                        results.append(f"✓ Power requirement: {comp_rules['power_requirement']}")
            
        elif validation_type == "pricing":
            # Validate pricing information
            if "product" in data and "price" in data:
                product = data["product"]
                price_str = data["price"]
                
                # Extract numeric price
                try:
                    price = float(price_str.replace("$", "").replace(",", ""))
                except:
                    issues.append("Invalid price format")
                    price = 0
                
                if product in validation_rules["pricing"]["price_validation"]:
                    price_range = validation_rules["pricing"]["price_validation"][product]
                    if price < price_range["min"]:
                        issues.append(f"Price ${price} is below expected range for {product}")
                    elif price > price_range["max"]:
                        issues.append(f"Price ${price} is above expected range for {product}")
                    else:
                        results.append(f"✓ Price ${price} is within expected range for {product}")
                
                # Check budget category
                for category, budget_range in validation_rules["pricing"]["budget_ranges"].items():
                    if budget_range["min"] <= price <= budget_range["max"]:
                        results.append(f"✓ Price falls in {category} budget category")
                        break
                        
        elif validation_type == "compatibility":
            # Validate component compatibility
            if "cpu" in data:
                cpu = data["cpu"]
                if cpu in validation_rules["compatibility"]:
                    comp_rules = validation_rules["compatibility"][cpu]
                    results.append(f"✓ {cpu} compatibility validated")
                    
                    if "ram" in data:
                        ram = data["ram"]
                        if ram < comp_rules["min_ram"]:
                            issues.append(f"RAM {ram} is insufficient for {cpu}")
                        elif ram > comp_rules["max_ram"]:
                            issues.append(f"RAM {ram} exceeds maximum for {cpu}")
                        else:
                            results.append(f"✓ RAM {ram} is compatible with {cpu}")
                    
                    if "storage" in data:
                        storage = data["storage"]
                        if storage not in comp_rules["compatible_storage"]:
                            recommendations.append(f"Consider {', '.join(comp_rules['compatible_storage'])} for optimal performance")
                        else:
                            results.append(f"✓ Storage {storage} is optimal for {cpu}")
                    
                    results.append(f"✓ Power requirement: {comp_rules['power_requirement']}")
                else:
                    issues.append(f"Unknown CPU: {cpu}")
                    
        elif validation_type == "requirements":
            # Validate against user requirements
            if "requirements" in data:
                requirements = data["requirements"]
                config = data.get("configuration", {})
                
                for req, value in requirements.items():
                    if req in config:
                        if config[req] >= value:
                            results.append(f"✓ Meets {req} requirement: {config[req]} >= {value}")
                        else:
                            issues.append(f"Does not meet {req} requirement: {config[req]} < {value}")
                    else:
                        issues.append(f"Cannot validate {req} requirement - not in configuration")
        
        # Compile final result
        if not issues and not results:
            return "No validation issues found, but no specific validation was performed."
        
        output = []
        if results:
            output.extend(results)
        
        if issues:
            output.append("\nIssues found:")
            for issue in issues:
                output.append(f"⚠ {issue}")
        
        if recommendations:
            output.append("\nRecommendations:")
            for rec in recommendations:
                output.append(f"💡 {rec}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error during validation: {str(e)}" 