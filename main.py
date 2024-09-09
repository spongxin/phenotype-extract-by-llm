from document import TxtDocument, XmlDocument
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from prompt import Prompt
from tqdm import tqdm
import argparse
import logging
import datetime
import client
import json
import os


logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--debug', '-d', action='store_true', help='print debug information in the console')
parser.add_argument('--config', '-c', type=str, default='config.ini', help='path to the configuration file')
parser.add_argument('--input', '-i', type=str, default=None, help='path to the input file')
args = parser.parse_args()

# Load configuration
config = ConfigParser()
config.read(args.config)

# Basic configuration
model_name = config['chat']['model'].split("/")[-1]
output_dir = os.path.join(config['settings']['output_path'], model_name)
history_dir = os.path.join(config['settings']['history_path'], model_name)

# Create output\history directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(history_dir):
    os.makedirs(history_dir)

# Load input files
input_path = config['settings']['input_path'] if args.input is None else args.input
endfix = config['settings']['input_file_endfix']
number_input_file = int(config['settings']['number_input_file'])
single_file = os.path.isfile(input_path)

# Select files by endfix and number_input_file
if not single_file:
    filenames = [i for i in os.listdir(input_path) if not endfix.strip() or i.endswith(endfix)]
    filenames = filenames[:number_input_file] if number_input_file > 0 else filenames

    # Drop files that already exist in the output directory
    if not config['settings']['overwrite']:
        number_input_file = len(filenames)
        fresh_filenames = [i for i in filenames if not os.path.exists(os.path.join(output_dir, i + '.json'))]
        number_droped_files = number_input_file - len(fresh_filenames)
        if number_droped_files > 0:
            logger.warning(f"droped {number_droped_files} files that already exist in the output directory.")


# Load prompts
prompts = Prompt(config['settings']['prompt_path']).prompts

# Load client
llm_config = config[client.ClientList[int(config['settings']['client_id'])].__name__]
llm_client = client.ClientList[int(config['settings']['client_id'])](**llm_config)


def extract(filename: str, **kwargs):
    """Extract phenotype information from XML file"""

    frames = ['None', ]
    try:
        Document = XmlDocument if filename.endswith('.xml') else TxtDocument
        doc = Document(filename, chunk_size=int(config['settings']['chunk_token_size']))
        logger.info(f"loaded {doc.name} at {datetime.datetime.now()}")
        paragraphs = doc.paragraph()
        chats = [
            {"role": "system", "content": prompts.get('fulltext-system-prompt')},
            {"role": "user", "content": ""}
        ]
        # Iterating
        for idx, paragraph in enumerate(paragraphs):
            content = prompts.get('fulltext-user-iter-prompt').replace("{{current_result}}", str(frames[-1])).replace("{{content}}", paragraph)
            chats[-1] = {"role": "user", "content": content}
            logger.info(f"extracting {idx + 1}th content of {doc.name} at {datetime.datetime.now()}")
            if args.debug:
                logger.debug(f"request content: \n{chats[-1]}\n")

            resp = llm_client.chat(messages=chats, **config['chat'])
            frames.append(resp.choices[0].message.content)
            logger.info(f"extracted {idx + 1}th content of {doc.name} at {datetime.datetime.now()}")
            if args.debug:
                logger.debug(f"lastest result: \n{frames[-1]}\n")
                input("Press `Enter` to continue...")
        # Reflecting
        for rf in range(int(config['settings']['max_reflect_time'])):
            chats[-1] = {"role": "user", "content": prompts.get('fulltext-user-finetune-prompt').replace("{{current_result}}", str(frames[-1]))}
            logger.info(f"{rf + 1}th reflecting {doc.name} at {datetime.datetime.now()}")
            resp = llm_client.chat(messages=chats, **config['chat'])
            frames.append(resp.choices[0].message.content)
            logger.info(f"{rf + 1}th reflected {doc.name} at {datetime.datetime.now()}")
            if args.debug:
                logger.debug(f"lastest result: \n{frames[-1]}\n")
                input("Press `Enter` to continue...")
        # Extract json data
        extracted = client.extract_json_data(frames[-1])
        fix_time = 0
        while type(extracted) == str and fix_time <= int(config['settings']['max_fix_time']):
            logger.error(extracted)
            chats[-1] = {"role": "user", "content": prompts.get('fulltext-user-fix-prompt').replace("{{current_result}}", str(frames[-1])).replace("{{error_message}}", extracted)}
            resp = llm_client.chat(messages=chats, **config['chat'])
            frames.append(resp.choices[0].message.content)
            extracted = client.extract_json_data(frames[-1])
            if args.debug:
                logger.debug(f"lastest result: \n{frames[-1]}\n")
                input("Press `Enter` to continue...")
            fix_time += 1
        # Save history
        with open(os.path.join(history_dir, doc.name+'.txt'), "w", encoding='utf-8') as f:
            f.write("\n\n".join(frames))
        logger.info(f"saved history to {os.path.join(output_dir, doc.name+'.txt')}")
        # Save output
        if type(extracted) == dict:
            with open(os.path.join(output_dir, doc.name+'.json'), "w", encoding='utf-8') as f:
                f.write(json.dumps(extracted, indent=4))
            logger.info(f"saved json output to {os.path.join(output_dir, doc.name+'.json')}")
        else:
            logger.warning(f"failed parse json from {doc.name}, see history for details at {os.path.join(history_dir, filename+'.pkl')}.")
        
    except Exception as e:
        logger.error(f"error occured when processing {filename}: \n{e}")
        if len(frames) > 1:
            with open(os.path.join(history_dir, doc.name+'.txt'), "w", encoding='utf-8') as f:
                f.write("\n\n".join(frames))
            logger.info(f"saved history to {os.path.join(output_dir, doc.name+'.txt')}")
        if args.debug:
            input("Press `Enter` to continue...")
        

def assign_tasks():
    if single_file:
        extract(input_path)
        return
    

    assert len(filenames) > 0
    pbar = tqdm(total=len(filenames), desc="processing tasks")
    update = lambda *args: pbar.update()
    with ThreadPoolExecutor(max_workers=config['settings']['thread_number']) as pool:
        for _, filename in enumerate(filenames):
            pool.submit(extract, os.path.join(input_path, filename)).add_done_callback(update)
            extract(filename)
    pbar.close()


if __name__ == '__main__':
    if args.debug:
        logging.basicConfig(level=logging.ERROR)
        logger.setLevel(level=logging.DEBUG)
    else:
        logging.basicConfig(filename='.log', level=logging.INFO)
        logger.setLevel(logging.INFO if args.verbose else logging.ERROR)

    assign_tasks()