import requests,json,os
import web_request as wr
from dotenv import load_dotenv
load_dotenv()
def search_github_issues(repo_name:str, q:str, per_page: int=2):
    url= "https://api.github.com/search/issues"
    params={"q": f"repo:{repo_name} {q}", "per_page": per_page, "sort": "created", "order": "desc"}
    return wr.robust_get(url, params=params)
results = search_github_issues("pallets/flask", "bug")
for item in results.get("items",[]):
    print("-", item["title"])