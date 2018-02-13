from bs4 import BeautifulSoup
import urllib.request
def crawl_webpage(url_in):
    with urllib.request.urlopen(url_in) as url:
        html = url.read()
        soup = BeautifulSoup(html, 'html.parser')
   
   # remove all script and style elements
    for script in soup(['script', 'style']):
        script.extract()  # rip it out
    
    p_tag_lists=''
    for p_tag in soup.findAll(['p','div']):
        t = p_tag.text.strip(' ').strip('\n')
        if(len(t) >= 250):
            print(len(t))
            print(t)
            p_tag_lists=p_tag_lists+'\n'+t
    # print('crawled words',p_tag_lists)
    return p_tag_lists

crawl_webpage('https://pantip.com/topic/37365102')