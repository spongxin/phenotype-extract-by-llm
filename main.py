from rich.progress import track
from document import Document
from prompt import Prompt
from groq import Groq
import argparse
import logging
import pickle
import time
import os


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='path to the directory containing XML files')
parser.add_argument('--prompt', '-p', type=str, default='prompts', help='path to the directory containing prompt files')
parser.add_argument('--output', '-o', type=str, default='output', help='path to the directory where to put the results(optional)')
parser.add_argument('--force', '-f', action='store_true', help='force re-processing XML input files when output files already exist')
parser.add_argument('--verbose', '-v', action='store_true', help='print information about processed files in the console')
parser.add_argument('--sleep', '-s', type=int, default=5, help='time to sleep between each request')
parser.add_argument('--model', '-m', type=str, default='llama3-70b-8192', help='model name to use for completion')
parser.add_argument('--history', '-t', type=str, default='.history', help='path to the directory containing chat history')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

# Create output\history directory
output_dir = os.path.join(args.output, args.model)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

history_dir = os.path.join(args.history, args.model)
if not os.path.exists(history_dir):
    os.makedirs(history_dir)


# Load XML files
filenames = [i for i in os.listdir(args.input) if i.endswith('.xml')]
if not args.force:
    filenames = [i for i in filenames if not os.path.exists(os.path.join(output_dir, i.replace('.xml', '.txt')))]
logging.warning(f"found {len(filenames)} files that need to be processed")

# Load prompts
prompts = Prompt(args.prompt).prompts


# Load API keys
with open("api-keys.txt", "r", encoding='utf-8') as f:
    clients = [Groq(api_key=key.strip()) for key in f.readlines() if key.strip()]
selected_client = -1
clients_num = len(clients)


# Get a client in order
def get_client():
    global selected_client
    selected_client = (selected_client + 1) % clients_num
    return clients[selected_client]


# Extract phenotype info from XML file
def extract(filename: str):
    results = [None, ]
    try:
        doc = Document(os.path.join(args.input, filename))
        paragraphs = doc.paragraph(min_length=1500)
        chats = [
            {"role": "system", "content": prompts.get('fulltext-system-prompt')},
            {"role": "user", "content": ""}
        ]
        for idx, paragraph in track(enumerate(paragraphs), description=f"extract {doc.name} by {args.model}"):
            content = prompts.get('fulltext-user-iter-prompt').replace("{{current_result}}", str(results[-1])).replace("{{content}}", paragraph)
            chats[-1] = {"role": "user", "content": content}
            logging.debug(f"request content: \n{content}\n")
            resp = get_client().chat.completions.create(
                model=args.model,
                messages=chats,
                temperature=0.1,
            )
            results.append(resp.choices[0].message.content)
            logging.debug(f"extracted {idx + 1} paragraphs from {doc.name}: \n{results[-1]}\n")
            if args.verbose:
                input(f"Press Enter to continue...")
            time.sleep(args.sleep)
        chats[-1] = {"role": "user", "content": prompts.get('fulltext-user-finetune-prompt') + str(results[-1])}
        resp = get_client().chat.completions.create(
            model=args.model,
            messages=chats,
            temperature=0.1,
        )
        results.append(resp.choices[0].message.content)
        logging.debug(f"finetuned output from {doc.name}: \n{results[-1]}\n")
        
        with open(os.path.join(output_dir, doc.name+'.txt'), "w", encoding='utf-8') as f:
            f.write(results[-1])
        logging.warning(f"saved output to {os.path.join(output_dir, doc.name+'.txt')}")
    
    except ValueError as e:
        logging.error(f"{filename}: {e}")
    
    finally:
        with open(os.path.join(history_dir, filename+'.pkl'), "wb") as f:
            pickle.dump(results, f)

for filename in filenames:
    extract(filename)