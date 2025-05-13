import arxiv
import time
import os
import json
from datetime import datetime

class ArxivDownloader:
    def __init__(self, download_dir="./papers", progress_file="download_progress.json"):
        self.download_dir = download_dir
        self.progress_file = progress_file
        self.downloaded_papers = self._load_progress()
        os.makedirs(download_dir, exist_ok=True)
    
    def _load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_progress(self, paper_id, title):
        self.downloaded_papers[paper_id] = {
            'title': title,
            'downloaded_at': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(self.downloaded_papers, f, indent=2)
    
    def download_papers(self, search_query, max_results=10, retry_attempts=3):
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        for paper in client.results(search):
            paper_id = paper.entry_id.split('/')[-1]
            
            # Пропускаем уже скачанные статьи
            if paper_id in self.downloaded_papers:
                print(f"Пропуск (уже скачано): {paper.title}")
                continue

            for attempt in range(retry_attempts):
                try:
                    print(f"Скачивание: {paper.title} (попытка {attempt + 1}/{retry_attempts})")
                    paper.download_pdf(dirpath=self.download_dir)
                    self._save_progress(paper_id, paper.title)
                    print(f"Успешно!")
                    time.sleep(4)  # Соблюдаем ограничения API
                    break
                except Exception as e:
                    print(f"Ошибка при скачивании {paper.title}: {e}")
                    if attempt < retry_attempts - 1:
                        wait_time = (attempt + 1) * 5  # Увеличиваем время ожидания с каждой попыткой
                        print(f"Повторная попытка через {wait_time} секунд...")
                        time.sleep(wait_time)
                    else:
                        print(f"Не удалось скачать после {retry_attempts} попыток")

# Пример использования
if __name__ == "__main__":
    downloader = ArxivDownloader()
    search_query = "machine learning AND cat:cs.LG AND submittedDate:[20240701 TO 20240801]"
    downloader.download_papers(search_query, max_results=1000) 