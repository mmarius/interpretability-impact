import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from utils import *
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

url = 'https://acl2018.org/programme/schedule/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')



papers = []
day_schedules = soup.find_all(class_='day-program')
for day in day_schedules:
    rows = day.find_all('tr')
    current_titles = []
    for row in rows:
        is_area_track_info_row = 'session-name-row' in row['class'] and 'conc-session-indiv-row' in row['class']
        is_orals_papers_row = 'conc-session-details-row' in row['class']
        is_poster_papers_row = 'poster-session-row' in row['class'] and len(list(row.children)) > 1
        if is_area_track_info_row:
            title_divs = row.find_all(class_='conc-session-name')
            title_regex = r'Session \d+[A-Za-z]*: (.*?)\s*\d*$'
            current_titles = [re.search(title_regex, div.text).group(1) for div in title_divs]
        elif is_orals_papers_row:
            papers_per_area = row.find_all('td')
            assert len(papers_per_area) == len(current_titles)
            for paper_column, area in zip(papers_per_area, current_titles):
                for paper_div in paper_column.find_all('div', class_='talk-title'):
                    paper_link_index = 2
                    links = paper_div.find_all('a')
                    paper_link = links[paper_link_index]
                    paper_id = paper_link['href'].split('/')[-1]
                    title = paper_link.text
                    papers.append({'id': paper_id, 'title': title, 'area': area})
            current_titles = []
        elif is_poster_papers_row:
            # print(row)
            poster_subsessions = row.find_all(class_='poster-sub-session')
            for subsession in poster_subsessions:
                raw_track_title = subsession.find(class_='poster-session-name').text
                if raw_track_title == 'Tutorial':
                    continue
                title_regex = r'Poster Session \d+[A-Za-z]*: (.*?)$'
                track = re.search(title_regex, raw_track_title).group(1)
                for paper_span in subsession.find_all('span'):
                    links = paper_span.find_all('a')
                    paper_link_index = 1
                    paper_link = links[paper_link_index]
                    paper_id = paper_link['href'].split('/')[-1]
                    title = paper_link.text
                    papers.append({'id': paper_id, 'title': title, 'area': track})
                    break

df = pd.DataFrame(papers)
print(df)

areas = set(list(df['area']))
for area in list(areas):
    print(area)
print(areas)
