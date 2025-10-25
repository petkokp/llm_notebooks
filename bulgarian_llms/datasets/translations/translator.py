import concurrent.futures
import requests
import re
import os
import html
import urllib.parse
import time  # NEW

class Translator:
    def __init__(self, source_language='auto', target_language='tr', timeout=5):
        self.source_language = source_language
        self.target_language = target_language
        self.timeout = timeout
        self.pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'

    def make_request(self, target_language, source_language, text, timeout):
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        url = 'https://translate.google.com/m?tl=%s&sl=%s&q=%s'%(target_language, source_language, escaped_text)

        # Minimal retry: try a few times if result is empty or a transient error happens
        for attempt in range(10):
            try:
                response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                result = response.text.encode('utf8').decode('utf8')
                result = re.findall(self.pattern, result)
                if result:
                    return html.unescape(result[0])
            except requests.RequestException:
                pass
            time.sleep(2 + attempt)  # small backoff: 1s, 2s, 3s

        # No more exits: raise so caller can decide
        raise RuntimeError('Error: translation failed or empty result after retries.')

    def translate(self, text, target_language='', source_language='', timeout=''):
        if text is None:
            return text
            
        if not target_language:
            target_language = self.target_language
        if not source_language:
            source_language = self.source_language
        if not timeout:
            timeout = self.timeout
        if len(text) > 5000:
            raise ValueError('Maximum 5000 characters at once are supported. (%d characters found.)'%(len(text)))
        if type(target_language) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.make_request, target, source_language, text, timeout) for target in target_language]
                return [f.result() for f in futures]
        return self.make_request(target_language, source_language, text, timeout)

    def translate_file(self, file_path, target_language='', source_language='', timeout=''):
        if not os.path.isfile(file_path):
            raise FileNotFoundError('The file or path is incorrect.')
        f = open(file_path, encoding='utf8')
        text = self.translate(f.read(), target_language, source_language, timeout)
        f.close()
        return text
