import json
import logging
import re
from datetime import datetime
import nltk
import spacy



import requests
from flask import jsonify, Flask, request

from configuration.datahub_prod import DATA_HUB_API, GRAPHQL_API_ENDPOINT, HEADERS, PLATFORMS, jwt_token,cookie_value

from openai import OpenAI
import os

from template.datahub_template import ddl_template,ddl_template_llm


from template.data_fetching_template import fetch_data

env = os.getenv('FLASK_ENV', 'development')
# if env == 'production':
from configuration.datahub_prod  import model_name, base_url, api_key
# else:
#     from config.llm_dev import llama_model, llama_base_url, llama_api_key
nlp = spacy.load("en_core_web_sm")

def fetch_all_search_entities(DATA_HUB_API, pod_name):
    """
    Fetch all dataset URNs based on pod name, and only include those with 'gold_layer' as a glossary term.

    :param DATA_HUB_API: API endpoint for fetching data.
    :param pod_name: The pod name to filter results (e.g., "oneapp", "onetv").
    :param Data_layer: Glossary term to filter for (e.g., "gold_layer").
    :return: A list of URNs.
    """
    url = f"{DATA_HUB_API}/entities?action=search"
    headers = {
        'X-RestLi-Protocol-Version': '2.0.0',
        'Content-Type': 'application/json'
    }

    data = {
        "entity": "dataset",
        "input": f"{pod_name} & tags:Live",
        "start": 0,
        "count": 700
    }

    urn_list = []

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            entities = response.json().get("value", {}).get("entities", [])

            for entity in entities:
                matched_fields = entity.get("matchedFields", [])

                for field in matched_fields:
                    if field.get("name") == "domains" and f"urn:li:domain:{pod_name}" in field.get("value", ""):
                        urn_list.append(entity["entity"])
                        break
        else:
            print("Error:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Error:", e)

    return urn_list


def fetch_glossary_terms(data_hub_api, urn):
    """
    Fetch glossary terms and their descriptions for a given dataset URN.
    """
    url = f"{data_hub_api}/api/graphql"
    headers = {
        'Content-Type': 'application/json',
    }
    query = """
    query {
        dataset(urn: "%s") {
            glossaryTerms {
                terms {
                    term {
                        urn
                        glossaryTermInfo {
                            name
                            description
                        }
                    }
                }
            }
        }
    }
    """ % urn

    data = {"query": query, "variables": {}}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()

            if not isinstance(response_json, dict):
                print(f"Unexpected response format for URN {urn}: {response_json}")
                return []

            dataset_data = response_json.get("data", {}).get("dataset", {})
            if not dataset_data:
                print(f"No dataset data found for URN {urn}. Response: {response_json}")
                return []

            glossary_terms_obj = dataset_data.get("glossaryTerms")
            if not glossary_terms_obj or not isinstance(glossary_terms_obj, dict):
                print(f"No glossary terms found for URN {urn}. Response: {response_json}")
                return []

            glossary_terms_data = glossary_terms_obj.get("terms", [])
            glossary_terms = []

            for term in glossary_terms_data:
                term_info = term.get("term", {}).get("glossaryTermInfo", {})
                name = term_info.get("name", "No glossary term")
                description = term_info.get("description", "No description available")

                if "gold layer" in name.lower() and "&nbsp;" in description:
                    description = description.split("&nbsp;")[0]

                if "bronze layer" in name.lower() and "&nbsp;" in description:
                    description = description.split("&nbsp;")[0]

                if "silver layer" in name.lower() and "&nbsp;" in description:
                    description = description.split("&nbsp;")[0]

                description = re.sub(r"&nbsp;", " ", description)
                description = re.sub(r"<[^>]+>", "", description)

                glossary_terms.append({
                    "name": name,
                    "glossary_term_description": description
                })
            return glossary_terms
        else:
            print(f"Failed to fetch glossary terms for URN {urn}. Status code:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching glossary terms for URN {urn}: {e}")
    except ValueError as e:
        print(f"Error parsing JSON response for URN {urn}: {e}")

    return []

# urn_list = fetch_all_search_entities(DATA_HUB_API,pod_name)
def filter_urns_by_gold_layer(data_hub_api, urn_list,data_layer):
    """
    Filter URNs to include only those that have 'gold_layer' in their glossary terms.
    """
    filtered_urns = []
    for urn in urn_list:
        glossary_terms = fetch_glossary_terms(data_hub_api, urn)
        if any(data_layer.lower() in term['name'].lower() for term in glossary_terms):
            filtered_urns.append(urn)
        else:
            print(f"Skipping {urn} - '{data_layer}' glossary term not found")
    return filtered_urns

def fetch_metadata_for_urn_list(urn_list, data_hub_api):
    """
    Fetch metadata for a list of URNs and extract relevant details.
    """
    metadata_list = []

    for table_urn in urn_list:
        url = f"{data_hub_api}/entities/{table_urn}"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            aspects = data['value'].get("com.linkedin.metadata.snapshot.DatasetSnapshot", {}).get('aspects', [])

            table_name = None
            table_desc = None
            field_descriptions = []
            owners = []
            glossary_terms = []
            schema_fields = {}
            editable_field_descriptions = {}

            for aspect in aspects:
                if 'com.linkedin.metadata.key.DatasetKey' in aspect:
                    table_name = aspect['com.linkedin.metadata.key.DatasetKey'].get('name')

                if 'com.linkedin.dataset.EditableDatasetProperties' in aspect:
                    table_desc = aspect['com.linkedin.dataset.EditableDatasetProperties'].get('description')

                if 'com.linkedin.schema.SchemaMetadata' in aspect:
                    fields = aspect['com.linkedin.schema.SchemaMetadata'].get('fields', [])
                    for field in fields:
                        field_name = field.get('fieldPath', '').split('.')[-1]
                        data_type = field.get('nativeDataType', 'UNKNOWN')
                        schema_fields[field_name] = data_type

                if 'com.linkedin.schema.EditableSchemaMetadata' in aspect:
                    editable_fields = aspect['com.linkedin.schema.EditableSchemaMetadata'].get('editableSchemaFieldInfo', [])
                    for editable_field in editable_fields:
                        field_path = editable_field.get('fieldPath', '').split('.')[-1]
                        description = editable_field.get('description', '')
                        editable_field_descriptions[field_path] = description

                if 'com.linkedin.common.Ownership' in aspect:
                    for owner in aspect['com.linkedin.common.Ownership'].get('owners', []):
                        owner_name = owner.get("owner", "").split(":")[-1] if owner.get("owner") else "Unknown"
                        owners.append({"name": owner_name, "type": owner.get("type", "UNKNOWN")})

            total_columns = len(schema_fields)
            described_columns = 0

            for field_name, data_type in schema_fields.items():
                description = editable_field_descriptions.get(field_name, "").strip()
                if description:
                    described_columns += 1
                else:
                    description = "No description available"
                formatted_description = f"{field_name} {data_type} - {description}"
                field_descriptions.append(formatted_description)

            metadata_list.append({
                "table_name": table_name,
                "table_description": table_desc,
                "field_descriptions": field_descriptions,
                "owners": owners,
                "total_columns": total_columns,
                "columns_with_descriptions": described_columns
            })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching metadata for URN {table_urn}: {e}")

    return metadata_list


def update_datahub_field_descriptions(
        urn: str,
        field_descriptions: dict,
        jwt_token: str,
        cookie_value: str
):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {jwt_token}',
        'Cookie': f'AWSALB={cookie_value}; AWSALBCORS={cookie_value}'
    }

    editable_schema_field_info = []
    for field_path, description in field_descriptions.items():
        editable_schema_field_info.append({
            "fieldPath": field_path,
            "description": description
            # Add "fieldDescription" if needed and supported
        })

    payload = {
        "query": """
        mutation updateDataset($urn: String!, $input: DatasetUpdateInput!) {
            updateDataset(urn: $urn, input: $input) {
                urn
            }
        }
        """,
        "variables": {
            "urn": urn,
            "input": {
                "editableSchemaMetadata": {
                    "editableSchemaFieldInfo": editable_schema_field_info
                }
            }
        }
    }

    response = requests.post(GRAPHQL_API_ENDPOINT, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        print("Update successful:", response.json())
    else:
        print("Update failed:", response.status_code, response.text)


Jargon_dic = {
    "NatCo": {
        "definition": "Refers to a national company or subsidiary within the DT Group that operates in a specific country within Europe",
        "values": ["[Country Code] (e.g., hr=Croatia, hu=Hungary)"]
    },
    "Platform": {
        "definition": "Digital environment where user interaction occurs",
        "values": [
            "Android (Mobile OS)",
            "iOS (Mobile OS)",
            "Web/Connected TV (OTT platforms)"
        ]
    },
    "Service Segment": {
        "definition": "Commercial Relationship Classification of a User",
        "values": [
            "B2B (Business-to-Business)",
            "B2C (Business-to-Consumer)"
        ]
    },
    "Service Category": {
        "definition": "Type of Service the user owns in Telekom",
        "values": [
            "Prepaid",
            "Postpaid",
            "Fixedline",
            "Converged (both mobile services and fixed line services, etc.)"
        ]
    },
    "hasall":{
        "definition": "The flag indicates whether the data that has been processed is considered final for that particular week, day, month, or the previous 30 days. If the value is 'false,' it means the date is not the month-ending date for the monthly runtype; otherwise, it will be 'true'",
        "values":[
            "true",
            "false"
         ]
    },
    "run_type":{
        "definition": "This field indicates whether aggregation has been performed on a daily, weekly, monthly, or previous 30-day basis",
        "values":[
            "daily",
            "weekly",
            "monthly",
            "previous 30-day"
        ]
    },
    "process_dt":{
        "definition": "  The date and time when the data was processed into this table."
    },
    "year":{
        "definition": "The year in which the data was collected or processed, providing a time context for the data."
    }

}

prompt1 = """
You are a data analyst with expertise in metadata creation. Your task is to generate clean and structured metadata for the provided table using the table description, DDL (schema definition), business glossary (jargon dictionary), and sample data.

IMPORTANT: 
- This metadata will be stored in a vector database and used in a Retrieval-Augmented Generation (RAG) system to support natural language queries.
- Use business language whenever possible (e.g., “customer”, “revenue”, “transaction” instead of “row”, “value”).
- Avoid repetition and generic terms.
- If a column's purpose is unclear from the name or DDL, use context clues from the sample data or glossary.
"""

prompt2 = """
You are a data analyst with expertise in metadata enhancement and semantic optimization. Your task is to generate clean, structured, and semantically rich metadata for the provided table. The metadata will be stored in a vector database and used in a Retrieval-Augmented Generation (RAG) system to support natural language queries.

Objectives:
    - Enhance existing technical descriptions from the DDL to make them business-friendly.
    - Generate new descriptions for columns with missing description.
    - Ensure all descriptions are concise, unambiguous, and rich in business meaning.
    - Use context from table description, schema (DDL), sample data, and business glossary.

Guidelines:
    - Use business language wherever possible.
    - Preserve the core technical meaning, but express it in clear, natural terms with business language.
    - Avoid repetition and vague phrases.

When no description is provided, infer the column's meaning using the column name, data type, sample values, and business glossary.
"""

def data_summarization(table_name: str, table_description: str, ddl: str, sample_data: str,prompt: str) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key)

    template = f"""
       {prompt}
**Input**

Table Name: {table_name}

Table Description:
{table_description}

Table Schema (DDL):
{ddl}

Sample Data:
{sample_data}

Business Glossary:
{str(Jargon_dic)}


**Output JSON Format**:
"""+"""
{
  "table_name": "",
  "columns": [
    {
      "column_name": "",
      "description": "",
      
    },
    {
      "column_name": "",
      "description": "",
      
    },
    ...for all column in ddl
  ]
}
"""

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": template}],
        temperature=0.1
    )

    response = completion.choices[0].message.content

    # Strip code block formatting if present
    if '```json' in response:
        response = response[(response.find('```json') + 7): response.rfind('```')]
    elif '```' in response:
        response = response[(response.find('```') + 3): response.rfind('```')]

    try:
        response_data = json.loads(response)
        parsed_table_name = response_data.get("table_name", "")
        columns = response_data.get("columns", [])

        print("Parsed Table Name:", parsed_table_name)
        for col in columns:
            col_name = col.get("column_name", "")
            desc = col.get("description", "")
            print(f"Column: {col_name}, Description: {desc}")

        return response_data

    except json.JSONDecodeError as e:
        print("JSON parsing failed. Here's the raw output:")
        print(response)
        return None


def fetch_tags_for_urn(urn: str, graphql_endpoint: str) -> list:
    query = """
    query {
      dataset(urn: "%s") {
        tags {
          tags {
            tag {
              urn
              name
              description
            }
          }
        }
      }
    }
    """ % urn

    response = requests.post(graphql_endpoint, json={"query": query})
    response.raise_for_status()
    data = response.json()

    tags = []
    try:
        tags = data["data"]["dataset"]["tags"]["tags"]
    except (KeyError, TypeError):
        logging.warning(f"No tags found for URN: {urn}")
    return [tag_obj["tag"]["name"] for tag_obj in tags if "tag" in tag_obj and "name" in tag_obj["tag"]]


def add_tag_to_urn(tag_name, urn, jwt_token, cookie_value):
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
        "Cookie": cookie_value
    }
    tag_urn = f"urn:li:tag:{tag_name}"
    payload = {
        "query": f"""
            mutation addTags {{
                addTags(input: {{
                    tagUrns: ["{tag_urn}"],
                    resourceUrn: "{urn}"
                }})
            }}
        """,
        "variables": {}
    }

    response = requests.post(GRAPHQL_API_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def parse_llm_output_to_field_descriptions(llm_output: dict) -> dict:
    field_descriptions = {}
    columns = llm_output.get("columns", [])
    for col in columns:
        column_name = col.get("column_name")
        description = col.get("description")
        if column_name and description:
            field_descriptions[column_name] = description
    return field_descriptions

app = Flask(__name__)

@app.route("/generate_metadata", methods=["POST"])
def generate_metadata():
    try:
        req_data = request.json
        pod_name = req_data.get("pod_name")
        data_layer = req_data.get("Data_layer")

        # Uncomment this validation if needed
        if not pod_name:
            return jsonify({
                "status": "Error",
                "message": "Missing required parameter: pod_name",
                "result_list": []
            }), 400

        urn_list = fetch_all_search_entities(DATA_HUB_API, pod_name)
        print(urn_list)

        urn=["urn:li:dataset:(urn:li:dataPlatform:tableau,2218bf01-714a-e0d3-59fb-d9a3722967d2,PROD)"]
        print(urn_list)
        filtered_urns = filter_urns_by_gold_layer(DATA_HUB_API, urn_list,data_layer)
        print("urn without gold_layer tag")
        print(filtered_urns)

        result_list = []

        def parse_field_description(field_descriptions):
            pattern = r'^(.+?)\s+(\w+)\s*-\s*(.+)$'
            result = {}
            for field in field_descriptions:
                match = re.match(pattern, field.strip())
                if match:
                    col_name, data_type, description = match.groups()
                    result[col_name.strip()] = description.strip()
                else:
                    print(f"Warning: Could not parse field description: {field}")
            return result

        for urn in filtered_urns:
            try:
                metadata_list = fetch_metadata_for_urn_list([urn], DATA_HUB_API)

                if metadata_list:
                    metadata = metadata_list[0]
                    full_table_name = metadata["table_name"]
                    table_name = full_table_name.replace("dt_analytics_aggregated.", "")
                    table_description = metadata["table_description"]
                    field_description = metadata["field_descriptions"]
                    column_description = metadata["columns_with_descriptions"]

                    ddl = fetch_data(f"SHOW TABLE {table_name};")
                    sample_data = fetch_data(f"SELECT * FROM {table_name} ORDER BY date DESC LIMIT 10 ;")

                    datahub_ddl = ddl_template(field_description, full_table_name)
                    datahub_dict = parse_field_description(field_description)

                    if column_description == 0:
                        llm_output = data_summarization(table_name, table_description, ddl, sample_data, prompt1)
                        tag_name = "mo_generated"
                    else:
                        llm_output = data_summarization(table_name, table_description, datahub_ddl, sample_data, prompt2)
                        tag_name = "mo_updated"

                    if llm_output is None or "columns" not in llm_output:
                        raise ValueError("LLM output was not successfully parsed")

                    tags = fetch_tags_for_urn(urn, GRAPHQL_API_ENDPOINT)


                    llm_dict = {col["column_name"]: col["description"] for col in llm_output["columns"]}
                    all_column_names = llm_dict.keys()
                    print(all_column_names)

                    merged_descriptions = []
                    for column in sorted(all_column_names):
                        if column.lower() == "create table":
                            continue
                        merged_descriptions.append({
                            "column": column,
                            "datahub_description": datahub_dict.get(column, ""),
                            "llm_description": llm_dict.get(column, "")
                        })

                    try:
                        add_tag_to_urn(tag_name, urn, jwt_token, cookie_value)
                    except Exception as tagging_error:
                        logging.warning(f"Tagging failed for {urn} with tag {tag_name}: {tagging_error}")

                    if "verified" not in tags:
                        field_descriptions_to_update = parse_llm_output_to_field_descriptions(llm_output)
                        update_datahub_field_descriptions(
                            urn=urn,
                            field_descriptions=field_descriptions_to_update,
                            jwt_token=jwt_token,
                            cookie_value=cookie_value
                        )
                    else:
                        print(f"Skipping update_datahub_field_descriptions for {urn} because 'mo_verified' tag is present.")

                    print("\n========================")
                    print(f"DataHub DDL for {full_table_name}:\n{datahub_ddl}")
                    print("------------------------")
                    print(f"LLM Output:\n{json.dumps(llm_output, indent=2)}")
                    print("========================\n")

                    for col_desc in merged_descriptions:
                        result_list.append({
                            "date": str(datetime.today().date()),
                            "table_name": full_table_name,
                            "column": col_desc["column"],
                            "datahub_description": col_desc["datahub_description"],
                            "llm_description": col_desc["llm_description"]
                        })

            except Exception as urn_exception:
                logging.exception(f"Error processing urn: {urn}")
                result_list.append({
                    "urn": urn,
                    "status": "Error",
                    "message": f"Failed to process urn: {urn}, error: {str(urn_exception)}"
                })

        if result_list:
            return jsonify({
                "status": "Success",
                "message": "Metadata generated successfully for all URNs.",
                "result_list": result_list
            })
        else:
            return jsonify({
                "status": "Error",
                "message": "No metadata found for the provided pod name.",
                "result_list": []
            }), 404

    except Exception as e:
        logging.exception("Unexpected error in generate_metadata")
        return jsonify({
            "status": "Error",
            "message": f"Internal server error: {str(e)}",
            "result_list": []
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)
