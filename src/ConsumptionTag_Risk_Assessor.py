import pandas as pd
import asyncio
import os
import time
import random
from tqdm import tqdm
import logging
import re
import json
from datetime import datetime, timedelta
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(""),
        logging.StreamHandler()
    ]
)


class DeepSeekRiskAssessor:
    """Consumption tag risk assessment system based on Silicon Flow OpenAI interface"""

    def __init__(self, api_key, model="", cache_file=""):
        self.api_key = api_key
        self.model = model
        self.cache_file = cache_file
        # Cache for tag→risk data
        self.tag_cache = self._load_cache()
        self.api_call_count = 0
        self.api_success_count = 0
        self.max_retries = 3


        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=""
        )


        self.semaphore = asyncio.Semaphore(50)
        self.total_tags = 0
        self.processed_tags = 0

    def _load_cache(self):

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"")
        return {}

    def _save_cache(self):

        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.tag_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"")

    def _parse_json_response(self, response_text):
        """Enhanced JSON parser to ensure risk_level is an integer"""
        try:
            # Try parsing original response directly
            data = json.loads(response_text)
            # Force convert risk_level to integer
            data["risk_level"] = int(data["risk_level"])
            return data
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logging.debug(f"")

        try:
            # Extract content within curly braces
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("")

            json_content = json_match.group(1)


            data = json.loads(json_content)

            data["risk_level"] = int(data.get("risk_level", 2))
            return data

        except Exception as e:
            logging.warning(f"")
            return {"risk_level": 2, "reason": f"Parsing failed, default risk level 2 (original response: {response_text[:50]}...)"}

    async def process_single_tag(self, tag):
        """Process single tag: check cache→call LLM→return risk data (ensure risk_level is integer)"""
        self.processed_tags += 1
        if self.total_tags > 0 and self.processed_tags % 10 == 0:
            logging.info(f"Tag processing progress: {self.processed_tags}/{self.total_tags}")

        # 1. Check cache first
        if tag in self.tag_cache:
            # Verify risk_level in cache is integer, fix abnormal values
            cache_data = self.tag_cache[tag]
            try:
                cache_data["risk_level"] = int(cache_data["risk_level"])
                if not (1 <= cache_data["risk_level"] <= 5):
                    raise ValueError("")
                return tag, cache_data
            except (TypeError, ValueError):
                logging.warning(f"Re-evaluating")


        for retry in range(self.max_retries):
            try:
                prompt = f"""
                # You can define your own risk level.

                """

                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=False,
                        max_tokens=200,
                        temperature=0.3
                    )

                self.api_call_count += 1
                response_text = response.choices[0].message.content.strip()
                risk_data = self._parse_json_response(response_text)

                # Verify risk level range
                if not (1 <= risk_data["risk_level"] <= 5):
                    raise ValueError(f"Risk level out of range: {risk_data['risk_level']}")

                # Update cache
                self.tag_cache[tag] = risk_data
                self._save_cache()
                self.api_success_count += 1
                logging.info(f"Added new tag to cache: {tag} (risk level: {risk_data['risk_level']})")
                return tag, risk_data

            except Exception as e:
                logging.warning(f"Tag {tag} failed on {retry + 1}st call: {str(e)[:50]}")
                if retry == self.max_retries - 1:

                    default_data = {"risk_level": 2, "reason": f"Evaluation failed, default risk level 2 (tag: {tag})"}
                    self.tag_cache[tag] = default_data
                    self._save_cache()
                    return tag, default_data
                await asyncio.sleep(1 + random.random())

    async def process_row_tags(self, tags_str):
        """Process all tags in a row (including total weighted score, total frequency, and final risk score)"""
        if not tags_str or str(tags_str).strip().lower() in ["nan", ""]:
            return {
                "risk_score": 2.0,
                "details": "Total weighted score: 0, Total frequency: 0, Final risk score: 2.0",
                "analysis": {}
            }

        tag_list = []
        for item in str(tags_str).split(';'):
            item = item.strip()
            if not item:
                continue

            if ':' in item:
                try:
                    tag_part, freq_part = item.split(':', 1)
                    tag = tag_part.strip()
                    freq = int(freq_part.strip())
                    if freq <= 0:
                        raise ValueError("Frequency must be positive")
                    tag_list.append({"tag": tag, "frequency": freq})
                except Exception as e:
                    logging.error(f"Failed to parse tag: {item} (error: {e})")
                    tag_list.append({"tag": item, "frequency": 1})
            else:
                tag_list.append({"tag": item, "frequency": 1})

        if not tag_list:
            return {
                "risk_score": 2.0,
                "details": "Total weighted score: 0, Total frequency: 0, Final risk score: 2.0",
                "analysis": {}
            }

        # Process risk levels for all tags
        self.total_tags = len(tag_list)
        self.processed_tags = 0
        tasks = [self.process_single_tag(item["tag"]) for item in tag_list]
        results = await asyncio.gather(*tasks)
        tag_risk_map = {tag: risk_data for tag, risk_data in results}

        total_weighted = 0  # Total weighted score
        total_freq = 0      # Total frequency
        analysis = {}
        for item in tag_list:
            tag = item["tag"]
            freq = item["frequency"]
            risk_data = tag_risk_map[tag]

            try:
                risk_level = int(risk_data["risk_level"])
                if not (1 <= risk_level <= 5):
                    raise ValueError("")
            except (TypeError, ValueError):
                logging.warning(f"")
                risk_level = 2

            weighted_score = risk_level * freq
            total_weighted += weighted_score  # Accumulate total weighted score
            total_freq += freq                # Accumulate total frequency

            analysis[tag] = {
                "risk_level": risk_level,
                "frequency": freq,
                "weighted_score": weighted_score,
                "reason": risk_data["reason"]
            }

        # Calculate final risk score
        risk_score = round(total_weighted / total_freq, 1) if total_freq > 0 else 2.0

        # Generate detail text (including total weighted score, total frequency, and final risk score)
        details = f"Total weighted score: {total_weighted}, Total frequency: {total_freq}, Final risk score: {risk_score}\n"
        details += "Analysis of each tag:\n"
        for tag, data in analysis.items():
            details += f"- {tag}: Risk level {data['risk_level']} (frequency {data['frequency']}), Reason: {data['reason'][:50]}\n"

        return {
            "risk_score": risk_score,
            "details": details.strip(),
            "analysis": analysis
        }

    async def update_excel(self, input_file, output_file=None):

        if not output_file:
            name, ext = os.path.splitext(input_file)
            output_file = f"{name}_full_analysis{ext}"
        else:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Read data
        df = pd.read_excel(input_file)
        required_cols = ['ConsumptionTags', 'Consumption risks', 'Details of Consumption Risks']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None  # Automatically create missing columns

        start_time = time.time()

        with tqdm(total=len(df), desc="Processing progress") as pbar:
            for idx, row in df.iterrows():
                tags_str = row['Consumption_Tags']
                try:
                    result = await self.process_row_tags(tags_str)
                    df.at[idx, 'Consumption risks'] = result["risk_score"]
                    df.at[idx, 'Details of Consumption Risks'] = result["details"]
                    logging.debug(f"Completed row {idx} analysis (number of tags: {len(result['analysis'])})")
                except Exception as e:
                    df.at[idx, 'Details of Consumption Risks'] = f"Processing failed: {str(e)[:50]}"
                    logging.error(f"Failed to process row {idx}: {e}")
                pbar.update(1)

        # Save results
        df.to_excel(output_file, index=False)
        total_time = time.time() - start_time
        logging.info(f"Full analysis completed! Results saved to: {output_file}")
        logging.info(f"Statistics: Total tags={self.total_tags}, API calls={self.api_call_count}, Successes={self.api_success_count}")
        logging.info(f"Time elapsed: {timedelta(seconds=int(total_time))}")


async def main():
    # Configuration parameters
    API_KEY = ""  # Valid API key
    INPUT_FILE = ""
    OUTPUT_FILE = ""
    CACHE_FILE = ""

    assessor = DeepSeekRiskAssessor(
        api_key=API_KEY,
        model="",
        cache_file=CACHE_FILE
    )
    await assessor.update_excel(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())