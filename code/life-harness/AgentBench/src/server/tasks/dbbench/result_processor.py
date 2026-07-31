from .interaction import Database

class DBResultProcessor:
    """
    Processes database query results and comparisons.
    Only exposes the compare_results and calculate_tables_hash interfaces.
    """
    
    @staticmethod
    def compare_results(answer, ground_truth, query_type):
        """
        Compare the answer with the ground truth.
        
        Args:
        answer - answer produced by the model
        ground_truth - ground-truth answer
        query_type - query type (SELECT/INSERT/UPDATE/DELETE)
        
        Returns:
        bool - whether the answer matches
        """
        try:
            # Process answer and ground_truth
            processed_answer = DBResultProcessor._clean_answer(answer)
            processed_ground_truth = DBResultProcessor._clean_answer(ground_truth)
            
            if query_type in ("INSERT", "DELETE", "UPDATE"):
                return processed_answer == processed_ground_truth
                
            # Print processed results for debugging
            print("Processed answer:", processed_answer)
            print("Processed ground_truth:", processed_ground_truth)
            
            # Comparison logic
            if len(processed_answer) == 1 and len(processed_ground_truth) == 1:
                # Get processed values
                ans_val = processed_answer[0]
                gt_val = processed_ground_truth[0]
                
                # Treat both special values (0, undefined, etc.) as equal
                if ans_val == "0" and gt_val == "0":
                    return True
                    
                # Float comparison
                if DBResultProcessor._is_float(ans_val) and DBResultProcessor._is_float(gt_val):
                    return DBResultProcessor._float_equal(ans_val, gt_val)

                # String comparison
                return ans_val == gt_val
            else:
                # If all values are floats, use float comparison
                if (all(DBResultProcessor._is_float(x) for x in processed_answer) and 
                    all(DBResultProcessor._is_float(x) for x in processed_ground_truth)):
                    # Check that each answer has a matching ground-truth value within tolerance
                    if len(processed_answer) != len(processed_ground_truth):
                        return False
                        
                    # Create match flags
                    matched_gt = [False] * len(processed_ground_truth)
                    
                    for ans in processed_answer:
                        matched = False
                        for i, gt in enumerate(processed_ground_truth):
                            if not matched_gt[i] and DBResultProcessor._float_equal(ans, gt):
                                matched_gt[i] = True
                                matched = True
                                break
                        if not matched:
                            return False
                            
                    return all(matched_gt)
                
                # Normal comparison using sets
                return set(processed_answer) == set(processed_ground_truth)
                    
        except Exception as e:
            print(f"Comparison error: {e}")
            return False
    
    @staticmethod
    async def calculate_tables_hash_async(database: Database, entry):
        """Asynchronously calculate the combined hash for all tables."""
        # Get table information, either a single table or a list of tables
        tables = entry["table"] if isinstance(entry["table"], list) else [entry["table"]]
        
        # Collect hashes for all tables
        table_hashes = []
        for table in tables:
            table_name = table["table_name"]
            table_info = table["table_info"]
            table_hash = await DBResultProcessor._get_table_hash_async(database, table_info, table_name)
            # Extract hash value
            cleaned_hash = table_hash.strip("[]()")
            hash_value = cleaned_hash.split(",")[0].strip().strip("'")
            table_hashes.append(hash_value)
        
        # Sort and combine all hash values
        combined_hash = "_".join(sorted(table_hashes))
        return combined_hash
    
    @staticmethod
    async def _get_table_hash_async(database: Database, table_info, table_name):
        """Asynchronously get the MD5 hash for a single table."""
        columns = ",".join(
            [f"`{column['name']}`" for column in table_info["columns"]]
        )
        md5_query = (
            f"select md5(group_concat(rowhash order by rowhash)) as hash "
            f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash "
            f"FROM `{table_name}`) as sub;"
        )
        return await database.execute(md5_query)
    
    @staticmethod
    def _normalize_special_values(value):
        """Handle special values, percentages, and formatted numbers."""
        if value is None:
            return "0"  # Convert None to "0"
        
        # Convert to string
        str_value = str(value).strip()
        
        # Handle percentages
        if str_value.endswith('%'):
            try:
                # Remove percent sign and keep numeric value
                return str_value[:-1].strip()
            except:
                pass
        
        # Handle thousands separators
        if ',' in str_value and not str_value.startswith('[') and not str_value.endswith(']'):
            try:
                # Remove commas
                str_value = str_value.replace(',', '')
            except:
                pass
        
        # Convert to lowercase for special-value comparison
        lower_value = str_value.lower()
        
        # Handle special-value mapping
        special_values_map = {
            "none": "0",
            "null": "0",
            "undefined": "0",
            "nan": "0",
            "inf": "0",
            "infinity": "0",
            "-inf": "0",
            "-infinity": "0",
            "": "0",  # Empty string
        }
        
        return special_values_map.get(lower_value, str_value)
    
    @staticmethod
    def _clean_mysql_result(result):
        """Handle special MySQL result formats such as [(value,)] or [(value1,), (value2,), ...]."""
        if isinstance(result, str) and result.startswith("[") and result.endswith("]"):
            try:
                # Try parsing with eval
                parsed_result = eval(result)
                
                # Check whether this is a list of tuples
                if isinstance(parsed_result, list) and all(isinstance(item, tuple) for item in parsed_result):
                    # Process each tuple
                    cleaned_values = []
                    for item in parsed_result:
                        # For a single-value tuple, (value,)
                        if len(item) == 1:
                            value = str(item[0]).strip().strip("'\"")
                            cleaned_values.append(value)

                    # Only return for single-col results. For multi-col tuples return None
                    # so the caller falls through to the generic string parser, which
                    # converts each tuple to its repr-string — matching what the agent submits.
                    return cleaned_values if cleaned_values else None
            except:
                # If eval fails, fall back to the original method
                pass
                
            # Original method for a single tuple
            try:
                # Handle formats like "[(293.0,)]"
                result_stripped = result.strip("[]")
                # Check whether there is only one tuple
                if result_stripped.count("(") == 1 and result_stripped.startswith("(") and result_stripped.endswith(",)"):
                    # Extract the value inside parentheses
                    value = result_stripped[1:-2]  # Remove "(" and ",)"
                    # Remove possible quotes
                    value = value.strip().strip("'\"")
                    return [value]
            except:
                pass
        return None
    
    
    @staticmethod
    def _clean_answer(answer):
        """Clean and normalize the answer."""
        # Handle None values
        if answer is None:
            return ["0"]
            
        # First check whether this is a MySQL result format
        mysql_result = DBResultProcessor._clean_mysql_result(answer)
        if mysql_result is not None:
            return [DBResultProcessor._normalize_special_values(x) for x in mysql_result]

        if isinstance(answer, str):
            # Remove extra whitespace
            answer = answer.strip()
            # If this is a list represented as a string
            if answer.startswith("[") and answer.endswith("]"):
                try:
                    # Try converting with eval first
                    cleaned = eval(answer)
                    if isinstance(cleaned, list):
                        # Handle possible tuple results
                        result = []
                        for item in cleaned:
                            if isinstance(item, tuple) and len(item) == 1:
                                # Handle tuple case, (value,)
                                value = str(item[0]).strip().strip("'\"")
                                result.append(DBResultProcessor._normalize_special_values(value))
                            else:
                                value = str(item).strip().strip("'\"")
                                result.append(DBResultProcessor._normalize_special_values(value))
                        return result
                except:
                    # If eval fails, parse manually
                    answer = answer[1:-1]
                    items = []
                    current = ""
                    in_quotes = False
                    for char in answer:
                        if char in '"\'':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            if current:
                                items.append(DBResultProcessor._normalize_special_values(current.strip().strip("'\"")))
                                current = ""
                        else:
                            current += char
                    if current:
                        items.append(DBResultProcessor._normalize_special_values(current.strip().strip("'\"")))
                    return items
            else:
                # Single value
                return [DBResultProcessor._normalize_special_values(answer.strip().strip("'\""))]
        elif isinstance(answer, (list, tuple)):
            # Handle list or tuple
            result = []
            for item in answer:
                if isinstance(item, tuple) and len(item) == 1:
                    # Handle tuple case, (value,)
                    value = str(item[0]).strip().strip("'\"")
                    result.append(DBResultProcessor._normalize_special_values(value))
                else:
                    value = str(item).strip().strip("'\"")
                    result.append(DBResultProcessor._normalize_special_values(value))
            return result
        else:
            return [DBResultProcessor._normalize_special_values(str(answer).strip().strip("'\""))]
    
    @staticmethod
    def _is_float(value):
        """Check whether the value can be converted to a float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def _float_equal(a, b, tol=1e-2):
        """Compare two floats with tolerance."""
        try:
            return abs(float(a) - float(b)) <= tol
        except (ValueError, TypeError):
            return False
