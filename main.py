"""
Extraktor v3.0 - Job Data Extraction Tool
Single-file version with all components integrated
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
import re
import csv
import json
import time
import threading
import psutil
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime


# ===========================
# CONFIGURATION & CONSTANTS
# ===========================

COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#2d2d2d',
    'bg_light': '#3e3e3e',
    'fg_primary': '#e0e0e0',
    'fg_secondary': '#a0a0a0',
    'accent_green': '#4ec9b0',
    'accent_blue': '#569cd6',
    'accent_orange': '#ce9178',
    'accent_red': '#f48771',
    'accent_yellow': '#dcdcaa',
}

SYSTEM_PROMPTS = {
    'job_extraction': """You are a job offer analyzer. Extract and structure information from job postings.
Focus on: job title, company, salary, requirements, contact information (emails, phone numbers, names).
Format your response as structured data.""",

    'summarization': """You are a text summarizer. Create concise, accurate summaries.
Focus on key points, requirements, and important details.
Keep summaries clear and actionable.""",

    'data_query': """You are a data analyst assistant. Answer questions about compiled job data.
Provide insights, comparisons, and recommendations based on the data provided.
Be precise and reference specific data points."""
}

MODEL_CONTEXTS = {
    'small': 2048,
    'medium': 4096,
    'large': 8192,
}

MODEL_RAM_USAGE = {
    'small': 512,
    'medium': 1536,
    'large': 4096,
}


# ===========================
# DATA STRUCTURES
# ===========================

@dataclass
class ModelInfo:
    """Information about an LM Studio model"""
    id: str
    name: str
    size_category: str
    context_size: int
    estimated_ram_mb: int
    is_loaded: bool = False

    def __str__(self):
        return f"{self.name} ({self.size_category}, {self.context_size} ctx, ~{self.estimated_ram_mb}MB)"


@dataclass
class JobPosting:
    """Structured job posting data"""
    url: str
    title: str = ""
    company: str = ""
    location: str = ""
    salary: str = ""
    description: str = ""
    requirements: str = ""
    emails: List[str] = None
    phone_numbers: List[str] = None
    contact_names: List[str] = None
    raw_text: str = ""
    scraped_at: float = 0.0

    def __post_init__(self):
        if self.emails is None:
            self.emails = []
        if self.phone_numbers is None:
            self.phone_numbers = []
        if self.contact_names is None:
            self.contact_names = []
        if self.scraped_at == 0.0:
            self.scraped_at = time.time()

    def to_dict(self):
        return asdict(self)

    def to_csv_row(self):
        return [
            self.title,
            self.company,
            self.location,
            self.salary,
            ", ".join(self.emails),
            ", ".join(self.phone_numbers),
            ", ".join(self.contact_names),
            self.url
        ]


# ===========================
# LM STUDIO MANAGER
# ===========================

class LMStudioManager:
    """Manages LM Studio connection, models, and operations"""

    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/v1/chat/completions"
        self.models_endpoint = f"{base_url}/v1/models"
        self.available_models: List[ModelInfo] = []
        self.current_model: Optional[ModelInfo] = None

    def test_connection(self) -> Dict:
        """Test connection to LM Studio server"""
        try:
            response = requests.get(self.models_endpoint, timeout=5)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Cannot connect to LM Studio. Is it running?"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def refresh_models(self) -> List[ModelInfo]:
        """Fetch and categorize available models from LM Studio"""
        self.available_models = []

        result = self.test_connection()
        if result["status"] != "success":
            return []

        models_data = result["data"].get("data", [])

        for model in models_data:
            model_id = model.get("id", "unknown")
            model_info = self._categorize_model(model_id)
            self.available_models.append(model_info)

        return self.available_models

    def _categorize_model(self, model_id: str) -> ModelInfo:
        """Categorize model based on its identifier"""
        model_lower = model_id.lower()

        if any(x in model_lower for x in ['350m', '0.6b', '600m']):
            category = 'small'
            context = MODEL_CONTEXTS['small']
            ram = MODEL_RAM_USAGE['small']
        elif any(x in model_lower for x in ['1.2b', '1.5b', '1b']):
            category = 'medium'
            context = MODEL_CONTEXTS['medium']
            ram = MODEL_RAM_USAGE['medium']
        elif any(x in model_lower for x in ['3b', '7b', '8b']):
            category = 'large'
            context = MODEL_CONTEXTS['large']
            ram = MODEL_RAM_USAGE['large']
        else:
            category = 'medium'
            context = MODEL_CONTEXTS['medium']
            ram = MODEL_RAM_USAGE['medium']

        name = model_id.replace('-', ' ').replace('_', ' ').title()

        return ModelInfo(
            id=model_id,
            name=name,
            size_category=category,
            context_size=context,
            estimated_ram_mb=ram
        )

    def select_model(self, model_id: str) -> bool:
        """Select a model for use"""
        for model in self.available_models:
            if model.id == model_id:
                self.current_model = model
                return True
        return False

    def get_system_ram_info(self) -> Dict:
        """Get current system RAM usage"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total // (1024 * 1024),
            'available_mb': memory.available // (1024 * 1024),
            'used_mb': memory.used // (1024 * 1024),
            'percent': memory.percent
        }

    def can_load_model(self, model: ModelInfo) -> tuple:
        """Check if system has enough RAM to load a model"""
        ram_info = self.get_system_ram_info()
        required_mb = model.estimated_ram_mb + 1024

        if ram_info['available_mb'] < required_mb:
            return False, f"Insufficient RAM. Need {required_mb}MB, have {ram_info['available_mb']}MB"

        if ram_info['percent'] > 85:
            return False, f"RAM usage too high ({ram_info['percent']}%). Close some applications."

        return True, "OK"

    def suggest_model_for_context(self, text_length: int) -> Optional[ModelInfo]:
        """Suggest appropriate model based on context size"""
        estimated_tokens = text_length // 4

        suitable_models = [
            m for m in self.available_models
            if m.context_size >= estimated_tokens
        ]

        if not suitable_models:
            return None

        suitable_models.sort(key=lambda x: MODEL_RAM_USAGE.get(x.size_category, 999999))
        return suitable_models[0]

    def send_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Send a prompt to LM Studio with progress tracking"""
        start_time = time.time()
        phases = []

        if model_id:
            model_to_use = model_id
        elif self.current_model:
            model_to_use = self.current_model.id
        else:
            return {"status": "error", "message": "No model selected"}

        total_text = prompt + (system_prompt or "")
        if self.current_model:
            suggested = self.suggest_model_for_context(len(total_text))
            if not suggested:
                return {
                    "status": "error",
                    "message": "Context too large for available models. Try summarizing first."
                }

            if suggested.id != self.current_model.id:
                if callback:
                    callback(
                        f"⚠ Context large. Recommending {suggested.name} instead of {self.current_model.name}",
                        "warning"
                    )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            if callback:
                callback("Preparing request", "preparation")
            phases.append({"phase": "preparation", "time": time.time() - start_time})

            if callback:
                callback(f"Sending to {model_to_use}", "sending")
            phases.append({"phase": "sending", "time": time.time() - start_time})

            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=180
            )

            if callback:
                callback("Model processing...", "processing")
            phases.append({"phase": "processing", "time": time.time() - start_time})

            if response.status_code == 200:
                data = response.json()
                end_time = time.time()

                if callback:
                    callback("Complete", "complete")
                phases.append({"phase": "complete", "time": end_time - start_time})

                return {
                    "status": "success",
                    "content": data["choices"][0]["message"]["content"],
                    "model": model_to_use,
                    "total_time": end_time - start_time,
                    "phases": phases,
                    "tokens": data.get("usage", {}),
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text[:200]}",
                    "total_time": time.time() - start_time,
                }

        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "message": "Request timed out (>180s). Try a smaller model or shorter prompt.",
                "total_time": time.time() - start_time,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "total_time": time.time() - start_time,
            }

    def chunk_and_summarize(
        self,
        long_text: str,
        chunk_size: int = 2000,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Break down long text into chunks and summarize with smaller model first"""
        if callback:
            callback("Text too large. Breaking into chunks...", "chunking")

        words = long_text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if callback:
            callback(f"Created {len(chunks)} chunks. Summarizing each...", "chunking")

        summaries = []
        small_models = [m for m in self.available_models if m.size_category == 'small']

        if not small_models:
            return {"status": "error", "message": "No small model available for chunking"}

        for i, chunk in enumerate(chunks, 1):
            if callback:
                callback(f"Summarizing chunk {i}/{len(chunks)}", "summarizing")

            result = self.send_prompt(
                chunk,
                system_prompt="Summarize this text concisely, keeping key information.",
                model_id=small_models[0].id,
                max_tokens=500
            )

            if result["status"] == "success":
                summaries.append(result["content"])
            else:
                summaries.append(f"[Error summarizing chunk {i}]")

        combined = "\n\n".join(summaries)

        return {
            "status": "success",
            "content": combined,
            "chunks_processed": len(chunks),
            "original_length": len(long_text),
            "summarized_length": len(combined)
        }


# ===========================
# JOB SCRAPER
# ===========================

class JobScraper:
    """Scrapes and extracts structured data from job postings"""

    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERNS = [
        r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
    ]
    NAME_PATTERN = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_job_url(self, url: str, callback: Optional[Callable] = None) -> JobPosting:
        """Scrape a single job posting URL"""
        job = JobPosting(url=url)

        try:
            if callback:
                callback(f"Fetching {url}", "fetching")

            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            if callback:
                callback("Parsing HTML", "parsing")

            soup = BeautifulSoup(response.content, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            job.title = self._extract_title(soup)
            job.company = self._extract_company(soup)
            job.raw_text = soup.get_text(separator=' ', strip=True)
            job.raw_text = ' '.join(job.raw_text.split())

            if callback:
                callback("Extracting contact information", "extraction")

            job.emails = self._extract_emails(job.raw_text)
            job.phone_numbers = self._extract_phone_numbers(job.raw_text)
            job.contact_names = self._extract_names(job.raw_text)
            job.salary = self._extract_salary(job.raw_text)
            job.location = self._extract_location(job.raw_text)

            if callback:
                callback("Extraction complete", "complete")

        except Exception as e:
            if callback:
                callback(f"Error: {str(e)}", "error")
            job.description = f"Error scraping: {str(e)}"

        return job

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for selector in ['h1', 'h2', '.job-title', '#job-title', '[class*="title"]']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        if soup.title:
            return soup.title.string.strip()
        return "Unknown Title"

    def _extract_company(self, soup: BeautifulSoup) -> str:
        for selector in ['.company', '#company', '[class*="company"]', '[class*="employer"]']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return "Unknown Company"

    def _extract_emails(self, text: str) -> List[str]:
        emails = re.findall(self.EMAIL_PATTERN, text)
        emails = list(set(emails))
        emails = [e for e in emails if not e.endswith('.png') and not e.endswith('.jpg')]
        return emails[:5]

    def _extract_phone_numbers(self, text: str) -> List[str]:
        phones = []
        for pattern in self.PHONE_PATTERNS:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        phones = list(set(phones))
        phones = [p.strip() for p in phones]
        return phones[:5]

    def _extract_names(self, text: str) -> List[str]:
        contact_patterns = [
            r'Contact:?\s+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Recruiter:?\s+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'HR:?\s+([A-Z][a-z]+ [A-Z][a-z]+)',
        ]

        names = []
        for pattern in contact_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)

        if not names:
            names = re.findall(self.NAME_PATTERN, text)

        names = list(set(names))
        false_positives = ['Privacy Policy', 'Terms Service', 'About Us', 'Contact Us']
        names = [n for n in names if n not in false_positives]

        return names[:3]

    def _extract_salary(self, text: str) -> str:
        salary_patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*-\s*\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?',
            r'\d{1,3}(?:,\d{3})*\s*(?:USD|EUR|CZK|Kč)',
            r'salary:?\s*([^\n]+)',
        ]

        for pattern in salary_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return "Not specified"

    def _extract_location(self, text: str) -> str:
        location_patterns = [
            r'Location:?\s*([^\n,]{3,50})',
            r'(?:Remote|Hybrid|On-site)',
            r'(?:Prague|Praha|Brno|Ostrava)',
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return "Not specified"

    def scrape_multiple_jobs(self, urls: List[str], callback: Optional[Callable] = None) -> List[JobPosting]:
        """Scrape multiple job URLs"""
        jobs = []

        for i, url in enumerate(urls, 1):
            if callback:
                callback(f"Scraping job {i}/{len(urls)}", "progress")

            job = self.scrape_job_url(url, callback)
            jobs.append(job)

            if i < len(urls):
                time.sleep(1)

        return jobs

    def filter_jobs(self, jobs: List[JobPosting], filters: Dict) -> List[JobPosting]:
        """Filter job postings based on criteria"""
        filtered = []

        for job in jobs:
            if filters.get('required_keywords'):
                text_lower = (job.title + " " + job.description + " " + job.raw_text).lower()
                if not any(kw.lower() in text_lower for kw in filters['required_keywords']):
                    continue

            if filters.get('excluded_keywords'):
                text_lower = (job.title + " " + job.description + " " + job.raw_text).lower()
                if any(kw.lower() in text_lower for kw in filters['excluded_keywords']):
                    continue

            if filters.get('remote_only'):
                if 'remote' not in job.location.lower() and 'remote' not in job.raw_text.lower():
                    continue

            filtered.append(job)

        return filtered


# ===========================
# DATA MANAGER
# ===========================

class DataManager:
    """Manages job data storage, export, and retrieval"""

    CSV_HEADERS = [
        "Title", "Company", "Location", "Salary", "Emails",
        "Phone Numbers", "Contact Names", "URL", "Scraped At"
    ]

    def __init__(self):
        self.jobs: List[JobPosting] = []

    def add_job(self, job: JobPosting):
        self.jobs.append(job)

    def add_jobs(self, jobs: List[JobPosting]):
        self.jobs.extend(jobs)

    def clear_all(self):
        self.jobs = []

    def get_all_jobs(self) -> List[JobPosting]:
        return self.jobs

    def get_job_count(self) -> int:
        return len(self.jobs)

    def export_to_csv(self, filepath: str) -> bool:
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

                for job in self.jobs:
                    row = [
                        job.title, job.company, job.location, job.salary,
                        ", ".join(job.emails), ", ".join(job.phone_numbers),
                        ", ".join(job.contact_names), job.url,
                        datetime.fromtimestamp(job.scraped_at).strftime("%Y-%m-%d %H:%M:%S")
                    ]
                    writer.writerow(row)
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

    def export_to_json(self, filepath: str) -> bool:
        try:
            data = [job.to_dict() for job in self.jobs]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def import_from_json(self, filepath: str) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                job = JobPosting(**item)
                self.jobs.append(job)
            return True
        except Exception as e:
            print(f"Error importing from JSON: {e}")
            return False

    def get_summary_stats(self) -> Dict:
        if not self.jobs:
            return {
                "total_jobs": 0,
                "jobs_with_salary": 0,
                "jobs_with_email": 0,
                "jobs_with_phone": 0,
                "unique_companies": 0,
                "unique_locations": 0,
            }

        return {
            "total_jobs": len(self.jobs),
            "jobs_with_salary": sum(1 for j in self.jobs if j.salary != "Not specified"),
            "jobs_with_email": sum(1 for j in self.jobs if j.emails),
            "jobs_with_phone": sum(1 for j in self.jobs if j.phone_numbers),
            "unique_companies": len(set(j.company for j in self.jobs)),
            "unique_locations": len(set(j.location for j in self.jobs)),
        }

    def search_jobs(self, query: str) -> List[JobPosting]:
        query_lower = query.lower()
        results = []

        for job in self.jobs:
            searchable = f"{job.title} {job.company} {job.description} {job.raw_text}".lower()
            if query_lower in searchable:
                results.append(job)

        return results

    def get_formatted_table_data(self) -> List[List[str]]:
        table_data = []
        for i, job in enumerate(self.jobs, 1):
            row = [
                str(i),
                job.title[:30] + "..." if len(job.title) > 30 else job.title,
                job.company[:20] + "..." if len(job.company) > 20 else job.company,
                job.location[:20] + "..." if len(job.location) > 20 else job.location,
                job.salary,
                str(len(job.emails)),
                str(len(job.phone_numbers))
            ]
            table_data.append(row)

        return table_data

    def compile_context_for_llm(self, max_jobs: int = 10) -> str:
        if not self.jobs:
            return "No job data available."

        jobs_to_include = self.jobs[:max_jobs]

        context = f"Job Data Summary ({len(self.jobs)} total jobs, showing {len(jobs_to_include)}):\n\n"

        for i, job in enumerate(jobs_to_include, 1):
            context += f"Job {i}:\n"
            context += f"  Title: {job.title}\n"
            context += f"  Company: {job.company}\n"
            context += f"  Location: {job.location}\n"
            context += f"  Salary: {job.salary}\n"
            if job.emails:
                context += f"  Emails: {', '.join(job.emails)}\n"
            if job.phone_numbers:
                context += f"  Phones: {', '.join(job.phone_numbers)}\n"
            context += f"  Description: {job.raw_text[:200]}...\n"
            context += "\n"

        if len(self.jobs) > max_jobs:
            context += f"\n(Plus {len(self.jobs) - max_jobs} more jobs not shown)\n"

        return context


# ===========================
# UI COMPONENTS
# ===========================

class ConsoleFrame(ttk.LabelFrame):
    """Compact console/log frame"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Console", **kwargs)

        self.text_widget = tk.Text(
            self, height=8, wrap=tk.WORD,
            bg=COLORS['bg_medium'], fg=COLORS['fg_primary'],
            insertbackground=COLORS['fg_primary'],
            selectbackground=COLORS['accent_blue'],
            font=("Consolas", 8), state=tk.DISABLED
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.text_widget.tag_configure('info', foreground=COLORS['accent_blue'])
        self.text_widget.tag_configure('success', foreground=COLORS['accent_green'])
        self.text_widget.tag_configure('warning', foreground=COLORS['accent_yellow'])
        self.text_widget.tag_configure('error', foreground=COLORS['accent_red'])
        self.text_widget.tag_configure('phase', foreground=COLORS['accent_orange'])

    def log(self, message: str, tag: str = 'info'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, f"[{timestamp}] ", 'info')
        self.text_widget.insert(tk.END, f"{message}\n", tag)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def clear(self):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.config(state=tk.DISABLED)


class PromptFrame(ttk.LabelFrame):
    """Prompt input frame with model selection"""

    def __init__(self, parent, on_send: Callable, **kwargs):
        super().__init__(parent, text="Prompt / Query", **kwargs)
        self.on_send = on_send

        model_row = ttk.Frame(self)
        model_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_row, text="Model:").pack(side=tk.LEFT, padx=(0, 5))

        self.model_var = tk.StringVar(value="No models loaded")
        self.model_dropdown = ttk.Combobox(
            model_row, textvariable=self.model_var,
            state='readonly', width=40
        )
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(model_row, text="Task:").pack(side=tk.LEFT, padx=(10, 5))

        self.task_var = tk.StringVar(value="job_extraction")
        self.task_dropdown = ttk.Combobox(
            model_row, textvariable=self.task_var,
            values=["job_extraction", "summarization", "data_query"],
            state='readonly', width=20
        )
        self.task_dropdown.pack(side=tk.LEFT)

        prompt_row = ttk.Frame(self)
        prompt_row.pack(fill=tk.X, padx=5, pady=5)

        self.prompt_entry = ttk.Entry(prompt_row, font=("Consolas", 10))
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.prompt_entry.bind('<Return>', lambda e: self.on_send())

        send_btn = ttk.Button(prompt_row, text="Send", command=self.on_send, width=10)
        send_btn.pack(side=tk.LEFT)

    def get_prompt(self) -> str:
        return self.prompt_entry.get().strip()

    def clear_prompt(self):
        self.prompt_entry.delete(0, tk.END)

    def get_selected_model(self) -> str:
        return self.model_var.get()

    def get_selected_task(self) -> str:
        return self.task_var.get()

    def update_models(self, models: List[str]):
        self.model_dropdown['values'] = models
        if models:
            self.model_var.set(models[0])


class DataTableFrame(ttk.LabelFrame):
    """Data table for displaying jobs"""

    def __init__(self, parent, on_row_select: Optional[Callable] = None, **kwargs):
        super().__init__(parent, text="Job Data", **kwargs)
        self.on_row_select = on_row_select

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("#", "Title", "Company", "Location", "Salary", "Emails", "Phones"),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            height=10
        )

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        self.tree.heading("#", text="#")
        self.tree.heading("Title", text="Title")
        self.tree.heading("Company", text="Company")
        self.tree.heading("Location", text="Location")
        self.tree.heading("Salary", text="Salary")
        self.tree.heading("Emails", text="Emails")
        self.tree.heading("Phones", text="Phones")

        self.tree.column("#", width=40, anchor=tk.CENTER)
        self.tree.column("Title", width=200)
        self.tree.column("Company", width=150)
        self.tree.column("Location", width=120)
        self.tree.column("Salary", width=100)
        self.tree.column("Emails", width=60, anchor=tk.CENTER)
        self.tree.column("Phones", width=60, anchor=tk.CENTER)

        self.tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.E, tk.W))

        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        if self.on_row_select:
            self.tree.bind('<<TreeviewSelect>>', self._on_select)

        stats_row = ttk.Frame(self)
        stats_row.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = ttk.Label(stats_row, text="No data loaded")
        self.stats_label.pack(side=tk.LEFT)

    def _on_select(self, event):
        selection = self.tree.selection()
        if selection and self.on_row_select:
            item = self.tree.item(selection[0])
            row_index = int(item['values'][0]) - 1
            self.on_row_select(row_index)

    def update_data(self, data: List[List[str]]):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in data:
            self.tree.insert("", tk.END, values=row)

    def update_stats(self, stats: dict):
        text = f"Total: {stats.get('total_jobs', 0)} | "
        text += f"With Salary: {stats.get('jobs_with_salary', 0)} | "
        text += f"With Email: {stats.get('jobs_with_email', 0)} | "
        text += f"Companies: {stats.get('unique_companies', 0)}"
        self.stats_label.config(text=text)

    def clear_data(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.stats_label.config(text="No data loaded")


class FilterFrame(ttk.LabelFrame):
    """Filter configuration"""

    def __init__(self, parent, on_apply: Callable, **kwargs):
        super().__init__(parent, text="Filters", **kwargs)
        self.on_apply = on_apply

        keyword_frame = ttk.Frame(self)
        keyword_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(keyword_frame, text="Required Keywords:").pack(anchor=tk.W)
        self.required_entry = ttk.Entry(keyword_frame)
        self.required_entry.pack(fill=tk.X, pady=2)

        ttk.Label(keyword_frame, text="Excluded Keywords:").pack(anchor=tk.W)
        self.excluded_entry = ttk.Entry(keyword_frame)
        self.excluded_entry.pack(fill=tk.X, pady=2)

        self.remote_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(keyword_frame, text="Remote only", variable=self.remote_var).pack(anchor=tk.W, pady=5)

        ttk.Button(keyword_frame, text="Apply Filters", command=self.on_apply).pack(pady=5)

    def get_filters(self) -> dict:
        return {
            'required_keywords': [kw.strip() for kw in self.required_entry.get().split(',') if kw.strip()],
            'excluded_keywords': [kw.strip() for kw in self.excluded_entry.get().split(',') if kw.strip()],
            'remote_only': self.remote_var.get()
        }


class ControlPanel(ttk.Frame):
    """Main control panel"""

    def __init__(self, parent, on_scrape, on_stop, on_export_csv, on_export_json, on_clear, **kwargs):
        super().__init__(parent, **kwargs)

        scrape_frame = ttk.LabelFrame(self, text="Scraping")
        scrape_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        self.scrape_btn = ttk.Button(scrape_frame, text="▶ Start Scraping", command=on_scrape, width=20)
        self.scrape_btn.pack(padx=5, pady=5)

        self.stop_btn = ttk.Button(scrape_frame, text="⏹ Stop", command=on_stop, state=tk.DISABLED, width=20)
        self.stop_btn.pack(padx=5, pady=5)

        export_frame = ttk.LabelFrame(self, text="Export")
        export_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        ttk.Button(export_frame, text="Export CSV", command=on_export_csv, width=15).pack(padx=5, pady=5)
        ttk.Button(export_frame, text="Export JSON", command=on_export_json, width=15).pack(padx=5, pady=5)

        data_frame = ttk.LabelFrame(self, text="Data")
        data_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        ttk.Button(data_frame, text="Clear All Data", command=on_clear, width=15).pack(padx=5, pady=5)

        self.status_label = ttk.Label(self, text="Status: Ready", foreground=COLORS['accent_green'])
        self.status_label.pack(side=tk.RIGHT, padx=20)

    def set_scraping_state(self, is_scraping: bool):
        if is_scraping:
            self.scrape_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Scraping...", foreground=COLORS['accent_orange'])
        else:
            self.scrape_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Ready", foreground=COLORS['accent_green'])

    def update_status(self, message: str, color: str = None):
        self.status_label.config(text=f"Status: {message}")
        if color:
            self.status_label.config(foreground=color)


# ===========================
# MAIN APPLICATION
# ===========================

class ExtractorApp:
    """Main application"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Extraktor v3.0 - Job Data Extraction Tool")
        self.root.geometry("1200x800")

        self.is_running = False
        self.job_urls: List[str] = []
        self.lm_studio_url = "http://localhost:1234"

        self.lm_manager = LMStudioManager(self.lm_studio_url)
        self.job_scraper = JobScraper()
        self.data_manager = DataManager()

        self._apply_dark_theme()
        self._create_menu()
        self._create_main_ui()

        self.root.after(1000, self._init_lm_studio)

    def _apply_dark_theme(self):
        self.root.configure(bg=COLORS['bg_dark'])

        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TFrame', background=COLORS['bg_dark'])
        style.configure('TLabel', background=COLORS['bg_dark'], foreground=COLORS['fg_primary'])
        style.configure('TLabelframe', background=COLORS['bg_dark'], foreground=COLORS['fg_primary'], bordercolor=COLORS['bg_light'])
        style.configure('TLabelframe.Label', background=COLORS['bg_dark'], foreground=COLORS['accent_blue'])
        style.configure('TButton', background=COLORS['bg_medium'], foreground=COLORS['fg_primary'], bordercolor=COLORS['bg_light'])
        style.map('TButton', background=[('active', COLORS['bg_light'])])
        style.configure('TEntry', fieldbackground=COLORS['bg_medium'], foreground=COLORS['fg_primary'])
        style.configure('TCombobox', fieldbackground=COLORS['bg_medium'], foreground=COLORS['fg_primary'], background=COLORS['bg_medium'])
        style.configure('Treeview', background=COLORS['bg_medium'], foreground=COLORS['fg_primary'], fieldbackground=COLORS['bg_medium'], bordercolor=COLORS['bg_light'])
        style.map('Treeview', background=[('selected', COLORS['accent_blue'])])

    def _create_menu(self):
        menubar = tk.Menu(self.root, bg=COLORS['bg_medium'], fg=COLORS['fg_primary'])
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_medium'], fg=COLORS['fg_primary'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import URLs...", command=self._import_urls)
        file_menu.add_command(label="Import JSON Data...", command=self._import_json_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        settings_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_medium'], fg=COLORS['fg_primary'])
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="LM Studio URL", command=self._configure_lm_studio)
        settings_menu.add_command(label="Manage Job URLs", command=self._manage_urls_dialog)

        tools_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_medium'], fg=COLORS['fg_primary'])
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Refresh Models", command=self._refresh_models)
        tools_menu.add_command(label="Test Connection", command=self._test_connection)
        tools_menu.add_command(label="Check RAM Usage", command=self._show_ram_info)

        help_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_medium'], fg=COLORS['fg_primary'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_ui(self):
        main_container = ttk.Frame(self.root, padding="5")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        self.console = ConsoleFrame(main_container, padding="5")
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        middle_frame = ttk.Frame(main_container)
        middle_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        middle_frame.columnconfigure(1, weight=3)
        middle_frame.rowconfigure(0, weight=1)

        self.filter_frame = FilterFrame(middle_frame, on_apply=self._apply_filters, padding="5")
        self.filter_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        self.data_table = DataTableFrame(middle_frame, on_row_select=self._on_job_selected, padding="5")
        self.data_table.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.prompt_frame = PromptFrame(main_container, on_send=self._send_prompt, padding="5")
        self.prompt_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.control_panel = ControlPanel(
            main_container,
            on_scrape=self._start_scraping,
            on_stop=self._stop_scraping,
            on_export_csv=self._export_csv,
            on_export_json=self._export_json,
            on_clear=self._clear_all_data,
            padding="5"
        )
        self.control_panel.grid(row=3, column=0, sticky=(tk.W, tk.E))

        self.console.log("Application initialized", 'success')

    def _init_lm_studio(self):
        self.console.log("Connecting to LM Studio...", 'info')

        def worker():
            result = self.lm_manager.test_connection()
            if result["status"] == "success":
                self.root.after(0, lambda: self.console.log("✓ Connected to LM Studio", 'success'))
                self.root.after(0, self._refresh_models)
            else:
                self.root.after(0, lambda: self.console.log(f"✗ LM Studio connection failed: {result['message']}", 'error'))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_models(self):
        self.console.log("Refreshing model list...", 'info')

        def worker():
            models = self.lm_manager.refresh_models()
            if models:
                model_names = [str(m) for m in models]
                self.root.after(0, lambda: self.prompt_frame.update_models(model_names))
                self.root.after(0, lambda: self.console.log(f"✓ Found {len(models)} models", 'success'))

                for model in models:
                    can_load, msg = self.lm_manager.can_load_model(model)
                    status = "✓" if can_load else "✗"
                    tag = 'success' if can_load else 'warning'
                    self.root.after(0, lambda m=model, s=status, t=tag: self.console.log(f"  {s} {m.name} - {m.estimated_ram_mb}MB", t))
            else:
                self.root.after(0, lambda: self.console.log("No models found. Load a model in LM Studio.", 'warning'))

        threading.Thread(target=worker, daemon=True).start()

    def _test_connection(self):
        self._init_lm_studio()

    def _show_ram_info(self):
        ram_info = self.lm_manager.get_system_ram_info()
        tag = 'warning' if ram_info['percent'] > 85 else 'success'
        self.console.log(f"RAM: {ram_info['used_mb']}/{ram_info['total_mb']}MB ({ram_info['percent']:.1f}%)", tag)

        message = f"""System RAM Information:

Total: {ram_info['total_mb']} MB
Used: {ram_info['used_mb']} MB ({ram_info['percent']:.1f}%)
Available: {ram_info['available_mb']} MB

Status: {'⚠ High usage' if ram_info['percent'] > 85 else '✓ OK'}"""
        messagebox.showinfo("RAM Usage", message)

    def _start_scraping(self):
        if not self.job_urls:
            messagebox.showwarning("No URLs", "Please add job URLs first (Settings > Manage Job URLs)")
            return

        self.is_running = True
        self.control_panel.set_scraping_state(True)
        self.console.log("=== STARTING JOB SCRAPING ===", 'success')

        def progress_callback(message: str, phase: str):
            self.root.after(0, lambda: self.console.log(f"  → {message}", 'phase'))

        def worker():
            scraped_jobs = []

            for i, url in enumerate(self.job_urls, 1):
                if not self.is_running:
                    self.root.after(0, lambda: self.console.log("Scraping stopped by user", 'warning'))
                    break

                self.root.after(0, lambda u=url, idx=i: self.console.log(f"[{idx}/{len(self.job_urls)}] Scraping: {u}", 'info'))

                job = self.job_scraper.scrape_job_url(url, progress_callback)
                scraped_jobs.append(job)

                self.root.after(0, lambda j=job: self.console.log(f"  ✓ {j.title} at {j.company}", 'success'))

            self.data_manager.add_jobs(scraped_jobs)
            self.root.after(0, self._update_data_display)

            if self.lm_manager.current_model and self.is_running:
                self.root.after(0, lambda: self._extract_with_ai(scraped_jobs))

            self.root.after(0, lambda: self.console.log(f"=== SCRAPING COMPLETE ({len(scraped_jobs)} jobs) ===", 'success'))
            self.root.after(0, lambda: self.control_panel.set_scraping_state(False))
            self.is_running = False

        threading.Thread(target=worker, daemon=True).start()

    def _stop_scraping(self):
        self.is_running = False
        self.console.log("Stopping scraping...", 'warning')

    def _extract_with_ai(self, jobs: List[JobPosting]):
        self.console.log("Running AI extraction on scraped jobs...", 'info')

        def worker():
            for i, job in enumerate(jobs[:5], 1):
                if not self.is_running:
                    break

                prompt = f"""Extract structured information from this job posting:

Title: {job.title}
Company: {job.company}
Text: {job.raw_text[:1000]}

Extract: requirements, benefits, contact info."""

                result = self.lm_manager.send_prompt(
                    prompt,
                    system_prompt=SYSTEM_PROMPTS['job_extraction'],
                    max_tokens=500,
                    callback=lambda msg, phase: self.root.after(0, lambda: self.console.log(f"  AI [{i}/5]: {msg}", 'phase'))
                )

                if result["status"] == "success":
                    job.description = result["content"]
                    self.root.after(0, lambda: self.console.log(f"  ✓ AI enhanced job {i}", 'success'))

        threading.Thread(target=worker, daemon=True).start()

    def _update_data_display(self):
        table_data = self.data_manager.get_formatted_table_data()
        self.data_table.update_data(table_data)
        stats = self.data_manager.get_summary_stats()
        self.data_table.update_stats(stats)

    def _apply_filters(self):
        filters = self.filter_frame.get_filters()
        self.console.log(f"Applying filters: {filters}", 'info')
        all_jobs = self.data_manager.get_all_jobs()
        filtered = self.job_scraper.filter_jobs(all_jobs, filters)
        self.console.log(f"Filtered to {len(filtered)} jobs", 'success')

    def _on_job_selected(self, index: int):
        jobs = self.data_manager.get_all_jobs()
        if 0 <= index < len(jobs):
            job = jobs[index]
            self.console.log(f"Selected: {job.title} at {job.company}", 'info')

    def _export_csv(self):
        if self.data_manager.get_job_count() == 0:
            messagebox.showwarning("No Data", "No data to export")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            success = self.data_manager.export_to_csv(filepath)
            if success:
                self.console.log(f"✓ Exported to {filepath}", 'success')
                messagebox.showinfo("Success", f"Data exported to {filepath}")
            else:
                self.console.log("✗ Export failed", 'error')

    def _export_json(self):
        if self.data_manager.get_job_count() == 0:
            messagebox.showwarning("No Data", "No data to export")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            success = self.data_manager.export_to_json(filepath)
            if success:
                self.console.log(f"✓ Exported to {filepath}", 'success')
                messagebox.showinfo("Success", f"Data exported to {filepath}")
            else:
                self.console.log("✗ Export failed", 'error')

    def _clear_all_data(self):
        if messagebox.askyesno("Confirm", "Clear all job data?"):
            self.data_manager.clear_all()
            self.data_table.clear_data()
            self.console.log("All data cleared", 'warning')

    def _send_prompt(self):
        prompt = self.prompt_frame.get_prompt()
        if not prompt:
            return

        task = self.prompt_frame.get_selected_task()
        self.console.log(f"Sending prompt (task: {task})...", 'info')
        self.prompt_frame.clear_prompt()

        if task == 'data_query':
            context = self.data_manager.compile_context_for_llm()
            full_prompt = f"{context}\n\nQuestion: {prompt}"

            if len(full_prompt) > 8000:
                self.console.log("⚠ Large context detected. Using summarization strategy...", 'warning')

                def worker():
                    result = self.lm_manager.chunk_and_summarize(
                        context,
                        callback=lambda msg, phase: self.root.after(0, lambda: self.console.log(f"  {msg}", 'phase'))
                    )

                    if result["status"] == "success":
                        summarized_prompt = f"{result['content']}\n\nQuestion: {prompt}"
                        self._send_to_lm_studio(summarized_prompt, task)

                threading.Thread(target=worker, daemon=True).start()
                return
            else:
                prompt = full_prompt

        self._send_to_lm_studio(prompt, task)

    def _send_to_lm_studio(self, prompt: str, task: str):
        def progress_callback(message: str, phase: str):
            self.root.after(0, lambda: self.console.log(f"  {message}", 'phase'))

        def worker():
            system_prompt = SYSTEM_PROMPTS.get(task, "")

            result = self.lm_manager.send_prompt(
                prompt,
                system_prompt=system_prompt,
                callback=progress_callback,
                max_tokens=1000
            )

            if result["status"] == "success":
                self.root.after(0, lambda: self.console.log(f"✓ Response ({result['total_time']:.2f}s):", 'success'))
                self.root.after(0, lambda: self.console.log(result['content'], 'info'))
            else:
                self.root.after(0, lambda: self.console.log(f"✗ Error: {result['message']}", 'error'))

        threading.Thread(target=worker, daemon=True).start()

    def _configure_lm_studio(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("LM Studio Configuration")
        dialog.geometry("400x150")
        dialog.configure(bg=COLORS['bg_dark'])

        ttk.Label(dialog, text="LM Studio URL:").pack(pady=10)

        url_var = tk.StringVar(value=self.lm_studio_url)
        url_entry = ttk.Entry(dialog, textvariable=url_var, width=40)
        url_entry.pack(pady=5)

        def save():
            self.lm_studio_url = url_var.get()
            self.lm_manager.base_url = url_var.get()
            self.lm_manager.api_endpoint = f"{url_var.get()}/v1/chat/completions"
            self.lm_manager.models_endpoint = f"{url_var.get()}/v1/models"
            self.console.log(f"LM Studio URL updated to {url_var.get()}", 'success')
            dialog.destroy()
            self._test_connection()

        ttk.Button(dialog, text="Save & Test", command=save).pack(pady=10)

    def _manage_urls_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Job URLs")
        dialog.geometry("600x400")
        dialog.configure(bg=COLORS['bg_dark'])

        ttk.Label(dialog, text="Job URLs (one per line):").pack(pady=10)

        text_widget = tk.Text(dialog, height=15, width=70, bg=COLORS['bg_medium'], fg=COLORS['fg_primary'])
        text_widget.pack(pady=10, padx=10)
        text_widget.insert(1.0, "\n".join(self.job_urls))

        def save():
            content = text_widget.get(1.0, tk.END).strip()
            self.job_urls = [url.strip() for url in content.split('\n') if url.strip()]
            self.console.log(f"Updated job URLs ({len(self.job_urls)} URLs)", 'success')
            dialog.destroy()

        ttk.Button(dialog, text="Save", command=save).pack(pady=10)

    def _import_urls(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
                self.job_urls.extend(urls)
                self.console.log(f"Imported {len(urls)} URLs", 'success')
            except Exception as e:
                self.console.log(f"Error importing URLs: {e}", 'error')

    def _import_json_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if filepath:
            success = self.data_manager.import_from_json(filepath)
            if success:
                self._update_data_display()
                self.console.log(f"Imported data from {filepath}", 'success')
            else:
                self.console.log("Import failed", 'error')

    def _show_about(self):
        about_text = """Extraktor v3.0

AI-powered job data extraction and analysis tool

Features:
• Web scraping from job sites
• AI-powered data extraction
• Smart model selection
• RAM-aware processing
• CSV/JSON export
• Advanced filtering

Developed with LM Studio integration"""
        messagebox.showinfo("About Extraktor", about_text)


def main():
    root = tk.Tk()
    app = ExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
