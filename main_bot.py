

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import os
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import json
import argparse

import requests
from requests.compat import urljoin

from dialogue_manager import DialogueManager
from utils import *

import logging


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=os.path.abspath('./bot.log'),
                    filemode='w+')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
logging.getLogger('').addHandler(console)

bot_logger = logging.getLogger('bot')

class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has tree main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager, master_name=None):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager
        
        

        
    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question, user_name=None):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        elif len(question) != len(question.encode()):
            return "Hmm, unseen characters ..."
        elif question.startswith('/'):
            return self.serve_master_commands(question)
        else:
            answer = self.dialogue_manager.generate_answer(question)
            
                           
            return answer
    
    def serve_master_commands(self, question):
        if question == '/report':
            return "I survived {} "
        elif question == '/snitch':
            return "Nada to read, Mate."
        else:
            return "Sorry, Mate! can't Comprehend"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--master", type=str, default=None)
    return parser.parse_args()

def is_unicode(text):
    return len(text) == len(text.encode())
	
def main():
    args = parse_args()
    
    token = args.token
    if not token and "TELEGRAM_TOKEN" in os.environ:
        token = os.environ["TELEGRAM_TOKEN"]
        
    if not token:
        bot_logger.error("Please, set bot token through --token or TELEGRAM_TOKEN env variable")
        return
        
    master = args.master
    if not token and "TELEGRAM_MASTER" in os.environ:
        master = os.environ["TELEGRAM_MASTER"] 
        
    if not master:
        bot_logger.warning("The Bot hasn't master."
                           "To use master commands,"
                           "put your username through --master or TELEGRAM_MASTER env variable.")

    dialogue_manager = DialogueManager(RESOURCE_PATH)
    bot = BotHandler(token, dialogue_manager, master_name=master)

    bot_logger.info("Ready to talk!")
    offset = 0
    while True:
        time.sleep(1)
        updates = bot.get_updates(offset=offset)
        for update in updates:
            bot_logger.info("Update content: {}".format(update))
            offset = max(offset, update["update_id"] + 1)
            
            if update.get("message", {}).get("text", None):
                chat_id = update["message"]["chat"]["id"]
                text = update["message"]["text"]
                
                answer = bot.get_answer(text)
                bot_logger.info("Answer: {}".format(answer))
                bot.send_message(chat_id, answer)
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        bot_logger.error(str(e))
        raise
