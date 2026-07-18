# here we call the JSONPlaceholder API to get a post with id 1 and print the response in a formatted way. We also print the status code, content type, and a preview of the response data.
# JSONPlaceholder is a free online REST API that you can use whenever you need some fake data. It can be used for testing and prototyping.
import requests, json

BASE = "https://jsonplaceholder.typicode.com"

# r= requests.get(f"{BASE}/posts/101", timeout=10) #we are getting the post with id 1 from the JSONPlaceholder API. We also set a timeout of 10 seconds to avoid hanging if the server is not responding.
# r.raise_for_status()
# data = r.json()
# print(json.dumps(data, indent=4))
# # print(type(data), data["id"], data["title"])
# print("Status-", r.status_code, "\n"
#     ,"Content-Type-", r.headers["Content-Type"], "\n"
#     ,"Preview-",json.dumps(data, indent=4)[:100]
#         )   


# query parameters and pagination patterns
# we are fetching posts from the JSONPlaceholder API in a paginated manner. We define a function fetch_posts_page that takes a page number and a page size as arguments. The function calculates the start and end indices for the posts to be fetched based on the page number and size. It then makes a GET request to the API to retrieve all posts, and returns the subset of posts corresponding to the requested page.

# def fetch_posts_page(page: int, size: int=10):
#     start = (page - 1) * size + 1
#     end = start + size - 1
#     r= requests. get(f"{BASE}/posts",timeout=10)
#     r.raise_for_status()    
#     all_posts = r.json()    
#     return all_posts[start-1:end]

# page1 = fetch_posts_page(1,5)
# page2 = fetch_posts_page(2,20)

# print("Page 1 len:", len(page1),"| first title:", page1[0]["title"])
# print("Page 2 len:", len(page2),"| first title:", page2[0]["title"])

# POST/PUT/DELETE
# we are creating a new post by sending a POST request to the JSONPlaceholder API. We define a new post as a dictionary with a title, body, and userId. We then send this data as JSON in the body of the POST request to the /posts endpoint. Finally, we print the response from the API, which includes the details of the created post.

# new_post = {"titel": "agentic AI", "body": "Mastering Agentic AI in 52 Weeks", "userId": 42}
# created= requests.post(f"{BASE}/posts", json=new_post, timeout=10).json()
# print("Created:", created)

# headers and authentication
# we are making an authenticated request to the GitHub API to check our rate limit status. We first load the GitHub token from the environment variables using the dotenv library. We then set up the headers for the request, including the Authorization header if the token is available. Finally, we make a GET request to the /rate_limit endpoint of the GitHub API and print the status code and rate limit information from the response.
import os
from dotenv import load_dotenv
load_dotenv()   

GIT_HUB_TOKEN = os.getenv("GIT_HUB_TOKEN")
headers = {"Accept":"application/vnd.github.v3+json"}

if GIT_HUB_TOKEN:
    headers["Authorization"] = f"token {GIT_HUB_TOKEN}"

# gh = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=10)
# print("GitHub status:", gh.status_code)
# print("Rate info:", gh.json().get("resources",{}).get("core",{}))


# we are fetching the 10 most recently closed issues from the Flask repository on GitHub. We define the owner and repo variables for the Flask repository, and then make a GET request to the GitHub API's /repos/{owner}/{repo}/issues endpoint with query parameters to filter for closed issues and limit the results to 10. We also include the necessary headers for authentication. Finally, we print the issue number and title for each of the retrieved issues.
# owner, repo = "pallets", "flask"
# issues = requests.get(f"https://api.github.com/repos/{owner}/{repo}/issues",
#                       headers=headers, params={"state":"closed","per_page":10}, timeout=10)
# issues.raise_for_status()
# for it in issues.json():
#     print(f"#{it['number']} - {it['title']}")



# we are defining a function called robust_get that performs a GET request to a specified URL with optional headers and query parameters. The function includes a retry mechanism that attempts the request up to a specified number of retries in case of failures, with an exponential backoff strategy to wait between attempts. If the request is successful, it returns the JSON response; otherwise, it raises an exception after exhausting all retries.
import time
 
def robust_get(url, headers=None, params=None, retries=3, backoff=1.5):
# this function performs a GET request to the specified URL with optional headers and query parameters. 
# It includes a retry mechanism that attempts the request up to a specified number of retries in case of failures, with an exponential backoff strategy to wait between attempts. If the request is successful, it returns the JSON response; otherwise, it raises an exception after exhausting all retries.
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 429:  # rate limited
                wait = int(r.headers.get("Retry-After", 2)) # if the server responds with a 429 status code, we check the "Retry-After" header to determine how long to wait before retrying. If the header is not present, we default to waiting for 2 seconds. We then sleep for the specified duration before continuing to the next attempt.
                time.sleep(wait); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e: # if any other request-related exception occurs (such as a connection error, timeout, or HTTP error), we catch the exception and print an error message that includes the attempt number and the exception details. If the current attempt is the last one (i.e., we've exhausted all retries), we re-raise the exception to signal that the operation has failed. Otherwise, we calculate the backoff time using an exponential strategy (increasing the wait time with each attempt) and sleep for that duration before retrying.
            print(f"[Attempt {attempt}] Error: {e}")
            if attempt == retries: raise
            time.sleep(backoff**attempt)
 
# data = robust_get(f"{BASE}/posts/2")
# print("Robust fetch title:", data["title"])

# 
def validate_post(obj:dict):
    required = {"userId","id","title","body"}
    missing = required - obj.keys()
    assert not missing, f"Missing keys: {missing}"
 
#validate_post(data)  # raises if shape changes

