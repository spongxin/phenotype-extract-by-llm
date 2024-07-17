from document import Document
from prompt import Prompt
from client import GroqClient
from tqdm import tqdm
import argparse
import logging
import pickle
import json
import os


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='path to the directory containing XML files')
parser.add_argument('--prompt', '-p', type=str, default='prompts', help='path to the directory containing prompt files')
parser.add_argument('--output', '-o', type=str, default='output', help='path to the directory where to put the results(optional)')
parser.add_argument('--force', '-f', action='store_true', help='force re-processing XML input files when output files already exist')
parser.add_argument('--verbose', '-v', action='store_true', help='print information about processed files in the console')
parser.add_argument('--sleep', '-s', type=int, default=60, help='time to sleep between each request')
parser.add_argument('--model', '-m', type=str, default='llama3-70b-8192', help='model name to use for completion. One of the [llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it, gemma2-9b-it]')
parser.add_argument('--history', type=str, default='.history', help='path to the directory containing chat history')
parser.add_argument('--num', '-n', type=int, default=-1, help='number of files to process')
parser.add_argument('--debug', '-d', action='store_true', help='print debug information in the console')
parser.add_argument('--finetune', '-ft', type=int, default=3, help='number of finetune iterations')
parser.add_argument('--length', '-l', type=int, default=2000, help='minimum length of the paragraph per request')
parser.add_argument('-thread', '-t', type=int, default=1, help='number of threads to use')
parser.add_argument('--host', '-hs', type=int, default=None, help='local host of llm service')
args = parser.parse_args()

logger = logging.getLogger(__name__)


# Create output\history directory
output_dir = os.path.join(args.output, args.model)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

history_dir = os.path.join(args.history, args.model)
if not os.path.exists(history_dir):
    os.makedirs(history_dir)


# Load Input files
filenames = [i for i in os.listdir(args.input) if i.endswith('.xml')]
if not args.force:
    filenames = [i for i in filenames if not os.path.exists(os.path.join(output_dir, i.replace('.xml', '.txt')))]
filenames = filenames[:args.num] if args.num > 0 else filenames

# Load prompts
prompts = Prompt(args.prompt).prompts


if args.host is None:
    # Groq client assigner
    GroqClient.interval_seconds = args.sleep
    clients = GroqClient()


def extract(filename: str, finetune: int, min_length: int, max_fix: int = 5):
    """
    Extract phenotype information from XML file
    
    :param filename: str, name of the XML file
    :param finetune: int, number of finetune iterations
    :param min_length: int, minimum length of the paragraph
    """
    results = [None, ]
    try:
        doc = Document(os.path.join(args.input, filename))
        paragraphs = doc.paragraph(min_length=min_length)
        chats = [
            {"role": "system", "content": prompts.get('fulltext-system-prompt')},
            {"role": "user", "content": ""}
        ]
        for idx, paragraph in enumerate(paragraphs):
            content = prompts.get('fulltext-user-iter-prompt').replace("{{current_result}}", str(results[-1])).replace("{{content}}", paragraph)
            chats[-1] = {"role": "user", "content": content}
            logger.debug(f"request content: \n{content}\n")
            resp = clients.get_aviliable_client().chat.completions.create(
                model=args.model,
                messages=chats,
                temperature=0.1,
            )
            results.append(resp.choices[0].message.content)
            logger.info(f"extracted {idx + 1} paragraphs from {doc.name}")
            if args.debug:
                logger.debug(f"\n{results[-1]}\n")
                input("Press `Enter` to continue iter...")
        chats = chats[:-1]
        for ft in range(finetune):
            chats.append({"role": "user", "content": prompts.get('fulltext-user-finetune-prompt') + str(results[-1])})
            resp = clients.get_aviliable_client().chat.completions.create(
                model=args.model,
                messages=chats,
                temperature=0.1,
            )
            results.append(resp.choices[0].message.content)
            chats.append({"role": "assistant", "content": resp.choices[0].message.content})
            logger.info(f"{ft} finetuned of {doc.name}")
            if args.debug:
                logger.debug(f"\n{results[-1]}\n")
                input("Press `Enter` to continue finetune...")
        with open(os.path.join(output_dir, doc.name+'.txt'), "w", encoding='utf-8') as f:
            f.write(results[-1])
        logger.info(f"saved text output to {os.path.join(output_dir, doc.name+'.txt')}")

        extracted = GroqClient.extract_json_data(results[-1])
        while type(extracted) == str and max_fix > 0:
            logger.warning(extracted)
            chats.append({"role": "user", "content": prompts.get('fulltext-user-fix-prompt').replace("{{current_result}}", str(results[-1])).replace("{{error_message}}", extracted)})
            resp = clients.get_aviliable_client().chat.completions.create(
                model=args.model,
                messages=chats,
                temperature=0.1,
            )
            results.append(resp.choices[0].message.content)
            chats.append({"role": "assistant", "content": resp.choices[0].message.content})
            extracted = GroqClient.extract_json_data(results[-1])
            if args.debug:
                logger.debug(f"\n{results[-1]}\n")
                input("Press `Enter` to continue fix...\n" + extracted)
            max_fix -= 1
        if type(extracted) == dict:
            with open(os.path.join(output_dir, doc.name+'.json'), "w", encoding='utf-8') as f:
                f.write(json.dumps(extracted, indent=4))
            logger.info(f"saved json output to {os.path.join(output_dir, doc.name+'.json')}")
        
    
    except Exception as e:
        logger.error(f"{filename}: {e}")
    
    finally:
        with open(os.path.join(history_dir, filename+'.pkl'), "wb") as f:
            pickle.dump(results, f)


def assign_multithread_tasks():
    from concurrent.futures import ThreadPoolExecutor
    if len(filenames) == 0:
        logger.warn("no files input.")
        return
    pbar = tqdm(total=len(filenames), desc="processing tasks")
    update = lambda *args: pbar.update()
    with ThreadPoolExecutor(max_workers=min(len(filenames), clients.clients_num)) as pool:
        for _, filename in enumerate(filenames):
            pool.submit(extract, filename, args.finetune, args.length).add_done_callback(update)
    pbar.close()


if __name__ == '__main__':
    logging.basicConfig(filename='.log', level=logging.ERROR)

    if args.debug:
        logging.basicConfig(level=logging.ERROR)
        logger.setLevel(level=logging.DEBUG)
        for idx, filename in enumerate(filenames):
            logger.info(f"[{idx}]Debug {filename}")
            extract(filename=filename, finetune=args.finetune, min_length=args.length)
    else:
        logger.setLevel(logging.INFO if args.verbose else logging.ERROR)
        assign_multithread_tasks()