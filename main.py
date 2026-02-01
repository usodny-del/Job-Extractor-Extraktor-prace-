"""
Extraktor v1.0 - Job Data Extraction Tool
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
    # MAIN APPLICATION
    # ===========================

    class ExtractorApp:
        """Main application"""

        def __init__(self, root: tk.Tk):
            # Set basic parameters of window
            self.root = root
            self.root.title("Extraktor v1.0 - Job Data Extraction Tool")
            self.root.geometry("1200x800")

            # Set up llms and app urls
            self.is_running = False
            self.job_urls: List[str] = []
            self.lm_studio_url = "http://localhost:1234"

            # Init other classes
            self.lm_manager = LMStudioManager(self.lm_studio_url)
            self.job_scraper = JobScraper()
            self.data_manager = DataManager()
            #Create UI
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
            style.configure('TLabelframe', background=COLORS['bg_dark'], foreground=COLORS['fg_primary'],
                            bordercolor=COLORS['bg_light'])
            style.configure('TLabelframe.Label', background=COLORS['bg_dark'], foreground=COLORS['accent_blue'])
            style.configure('TButton', background=COLORS['bg_medium'], foreground=COLORS['fg_primary'],
                            bordercolor=COLORS['bg_light'])
            style.map('TButton', background=[('active', COLORS['bg_light'])])
            style.configure('TEntry', fieldbackground=COLORS['bg_medium'], foreground=COLORS['fg_primary'])
            style.configure('TCombobox', fieldbackground=COLORS['bg_medium'], foreground=COLORS['fg_primary'],
                            background=COLORS['bg_medium'])
            style.configure('Treeview', background=COLORS['bg_medium'], foreground=COLORS['fg_primary'],
                            fieldbackground=COLORS['bg_medium'], bordercolor=COLORS['bg_light'])
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
                    self.root.after(0, lambda: self.console.log(f"✗ LM Studio connection failed: {result['message']}",
                                                                'error'))

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
                        self.root.after(0, lambda m=model, s=status, t=tag: self.console.log(
                            f"  {s} {m.name} - {m.estimated_ram_mb}MB", t))
                else:
                    self.root.after(0,
                                    lambda: self.console.log("No models found. Load a model in LM Studio.", 'warning'))

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

                    self.root.after(0,
                                    lambda u=url, idx=i: self.console.log(f"[{idx}/{len(self.job_urls)}] Scraping: {u}",
                                                                          'info'))

                    job = self.job_scraper.scrape_job_url(url, progress_callback)
                    scraped_jobs.append(job)

                    self.root.after(0, lambda j=job: self.console.log(f"  ✓ {j.title} at {j.company}", 'success'))

                self.data_manager.add_jobs(scraped_jobs)
                self.root.after(0, self._update_data_display)

                if self.lm_manager.current_model and self.is_running:
                    self.root.after(0, lambda: self._extract_with_ai(scraped_jobs))

                self.root.after(0, lambda: self.console.log(f"=== SCRAPING COMPLETE ({len(scraped_jobs)} jobs) ===",
                                                            'success'))
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
                        callback=lambda msg, phase: self.root.after(0, lambda: self.console.log(f"  AI [{i}/5]: {msg}",
                                                                                                'phase'))
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
                            callback=lambda msg, phase: self.root.after(0,
                                                                        lambda: self.console.log(f"  {msg}", 'phase'))
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
                    self.root.after(0,
                                    lambda: self.console.log(f"✓ Response ({result['total_time']:.2f}s):", 'success'))
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
            about_text = """Extraktor v1.0

    AI-pohanený analyzační nastroj na hledání práce

    Funkce:
    • Automatické scrapování prácí
    • AI-Poháněná extrakce dat
    • Chytré 
    • RAM-aware processing
    • CSV/JSON export
    • Advanced filtering

    Developed with LM Studio integration"""
            messagebox.showinfo("About Extraktor", about_text)







#Entry point
    def main():
        root = tk.Tk()
        app = ExtractorApp(root)
        root.mainloop()

    if __name__ == "__main__":
        main()
