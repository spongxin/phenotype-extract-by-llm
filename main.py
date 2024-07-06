from document import Document
from prompt import Prompt
from client import Client
from tqdm import tqdm
import argparse
import logging
import pickle
import os


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='path to the directory containing XML files')
parser.add_argument('--prompt', '-p', type=str, default='prompts', help='path to the directory containing prompt files')
parser.add_argument('--output', '-o', type=str, default='output', help='path to the directory where to put the results(optional)')
parser.add_argument('--force', '-f', action='store_true', help='force re-processing XML input files when output files already exist')
parser.add_argument('--verbose', '-v', action='store_true', help='print information about processed files in the console')
parser.add_argument('--sleep', '-s', type=int, default=30, help='time to sleep between each request')
parser.add_argument('--model', '-m', type=str, default='llama3-70b-8192', help='model name to use for completion')
parser.add_argument('--history', '-t', type=str, default='.history', help='path to the directory containing chat history')
parser.add_argument('--num', '-n', type=int, default=-1, help='number of files to process')
parser.add_argument('--debug', '-d', action='store_true', help='print debug information in the console')
parser.add_argument('--finetune', '-ft', type=int, default=3, help='number of finetune iterations')
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
filenames = filenames[:args.num]

# Load prompts
prompts = Prompt(args.prompt).prompts


# Groq client assigner
Client.interval_seconds = args.sleep
clients = Client()


def extract(filename: str, finetune: int, min_length: int = 3000):
    """
    Extract phenotype information from XML file
    
    :param filename: str, name of the XML file
    :param api_key: str, OpenAI API key
    :param min_length: int, minimum length of the paragraph
    :param finetune: int, number of finetune iterations
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
                input("Press Enter to continue...")
        for ft in range(finetune):
            chats[-1] = {"role": "user", "content": prompts.get('fulltext-user-finetune-prompt') + str(results[-1])}
            resp = clients.get_aviliable_client().chat.completions.create(
                model=args.model,
                messages=chats,
                temperature=0.1,
            )
            results.append(resp.choices[0].message.content)
            logger.info(f"{ft} finetuned of {doc.name}")
            if args.debug:
                logger.debug(f"\n{results[-1]}\n")
                input("Press Enter to continue...")
        with open(os.path.join(output_dir, doc.name+'.txt'), "w", encoding='utf-8') as f:
            f.write(results[-1])
        logger.info(f"saved output to {os.path.join(output_dir, doc.name+'.txt')}")
    
    except Exception as e:
        logger.error(f"{filename}: {e}")
    
    finally:
        with open(os.path.join(history_dir, filename+'.pkl'), "wb") as f:
            pickle.dump(results, f)


def assign_multithread_tasks():
    from concurrent.futures import ThreadPoolExecutor
    if len(filenames) == 0:
        return
    pbar = tqdm(total=len(filenames), desc="processing tasks")
    update = lambda *args: pbar.update()
    with ThreadPoolExecutor(max_workers=min(len(filenames), clients.clients_num)) as pool:
        for _, filename in enumerate(filenames):
            pool.submit(extract, filename, args.finetune).add_done_callback(update)


if __name__ == '__main__':
    logging.basicConfig(filename='.log', level=logging.INFO if args.verbose else logging.ERROR)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    assign_multithread_tasks()