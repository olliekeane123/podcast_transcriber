
import requests
from requests import HTTPError
from bs4 import BeautifulSoup
import datetime
import hashlib

from src.logger import init_logger
logger = init_logger(__name__)


### Request to get RSS text ###

def fetch_rss_feed():
    rss_url = "https://anchor.fm/s/fd1fcb44/podcast/rss"

    try:
        response = requests.get(rss_url)
        response.raise_for_status()

        logger.info("RSS feed fetched successfully")
        return response.text
    except HTTPError as err:
        logger.error(f"HTTP error occured when fetching RSS: {err}")
    except Exception as err:
        logger.error(f"Unexpected error occured when fetching RSS: {err}")


### Extract list of episodes from RSS text ###

def get_ep_xml_list(xml_text: str):
    xml_soup = BeautifulSoup(xml_text, "xml")
    episodes = xml_soup.find_all("item")

    if not episodes:
        logger.error("Could not parse episode entries")

    return episodes


### Extract all episode data using helper functions ###

def get_ep_metadata(ep_xml: BeautifulSoup):
    try:
        title = get_ep_title(ep_xml)

        audio_url = get_ep_audio_url(ep_xml, title)
        if not audio_url:
            logger.warning(f"Skipping episode '{title}': Missing audio URL")
            return None
        
        episode_id = hashlib.md5(audio_url.encode()).hexdigest()[:12]
        pub_date = get_ep_pub_date(ep_xml, title)
        description = get_ep_descripton(ep_xml, title)

        return {
            "episode_id": episode_id,
            "title": title,
            "audio_url": audio_url,
            "pub_date": pub_date,
            "description": description,
            "ingested_at": datetime.datetime.now().isoformat()
        }
    except Exception as err:
        logger.error(f"Unable to retrieve metadata: {err}")

    
### Helper functions to extract specific date - title, pub_date, description and audio_url ###

def get_ep_title(xml: BeautifulSoup):
    title_tag = xml.find("title")

    title = title_tag.text if title_tag else "untitled"

    return title

def get_ep_pub_date(xml: BeautifulSoup, title: str):
    pub_date_tag = xml.find("pubDate")

    pub_date = pub_date_tag.text if pub_date_tag else None

    if not pub_date:
        logger.warning(f"Could not find pub_date for episode: {title}")

    return pub_date

def get_ep_descripton(xml: BeautifulSoup, title: str):
    description_tag = xml.find("description")
    
    if not description_tag:
        logger.warning(f"Could not find a description for episode: {title}")
        return ""

    raw_html = description_tag.text 

    inner_soup = BeautifulSoup(raw_html, "html.parser")

    paragraphs = inner_soup.find_all("p")
    
    if paragraphs:
        clean_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        return clean_text

    return inner_soup.get_text(strip=True)

def get_ep_audio_url(xml: BeautifulSoup, title: str):
    enclosure = xml.find("enclosure")

    if not enclosure and enclosure.has_attr("url"):
        logger.error(f"Could not find audio URL for episode: {title}")
        return None
        
    return enclosure["url"]
    

### Extract audio data from url ###

def stream_audio(audio_url: str):
    try:
        audio_response = requests.get(audio_url, stream=True)
        audio_response.raise_for_status()

        logger.info("Episode audio fetched successfully")
        return audio_response
    except HTTPError as err:
        logger.error(f"HTTP error occured when fetching episode audio: {err}")
    except Exception as err:
        logger.error(f"Unexpected error occured when fetching episode audio: {err}")
