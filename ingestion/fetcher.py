import os
import shutil
import tempfile
from git import Repo
from urllib.parse import urlparse
from utils.logger import get_logger

logger = get_logger(__name__)

class RepoFetcher:
    """Fetches a GitHub repository to a local directory for ingestion."""
    
    def __init__(self, cache_dir: str = "data/repo_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def fetch(self, repo_url: str) -> str:
        """
        Clones a GitHub repository to a local directory.
        If it already exists in the cache, returns the cached path (or pulls latest).
        
        Args:
            repo_url: URL of the GitHub repository (e.g., https://github.com/user/repo)
            
        Returns:
            The absolute path to the local repository directory.
        """
        parsed = urlparse(repo_url)
        if not parsed.netloc:
            # Assume it's already a local path
            if os.path.exists(repo_url):
                logger.info(f"Using local path directly: {repo_url}")
                return repo_url
            else:
                raise ValueError(f"Invalid URL or local path not found: {repo_url}")
                
        # Generate a safe directory name from the URL
        # e.g., github.com/user/repo -> github.com_user_repo
        safe_name = f"{parsed.netloc}{parsed.path}".replace("/", "_").replace("\\", "_")
        local_path = os.path.abspath(os.path.join(self.cache_dir, safe_name))
        
        if os.path.exists(local_path):
            logger.info(f"Repository already exists in cache at {local_path}. Pulling latest...")
            try:
                repo = Repo(local_path)
                origin = repo.remotes.origin
                origin.pull()
                logger.info("Successfully pulled latest changes.")
            except Exception as e:
                logger.warning(f"Failed to pull latest changes: {e}. Using existing cached version.")
        else:
            logger.info(f"Cloning {repo_url} into {local_path}...")
            try:
                Repo.clone_from(repo_url, local_path)
                logger.info("Clone complete.")
            except Exception as e:
                logger.error(f"Failed to clone repository: {e}")
                raise
                
        return local_path
        
    def cleanup(self):
        """Optionally remove the cloned repositories."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleaned up repository cache at {self.cache_dir}")
