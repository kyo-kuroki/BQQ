import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm





def get_wikitext2_trainloader(nsamples, seed, seqlen, model, tokenizer=None, batch_size=1, shuffle=True):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    import random

    # データ読み込み
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # トークナイザ読み込み
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # テキストをトークナイズ
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    input_ids = trainenc.input_ids
    total_len = input_ids.shape[1]
    num_chunks = total_len // seqlen

    # サンプリングロジック
    if nsamples is None or nsamples >= num_chunks:
        indices = range(num_chunks)
    else:
        random.seed(seed)
        indices = random.sample(range(num_chunks), nsamples)

    # チャンク分割してテンソル結合
    inps, tars = [], []
    for i in indices:
        start = i * seqlen
        end = start + seqlen
        inp = input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        inps.append(inp)
        tars.append(tar)

    inps = torch.cat(inps, dim=0)
    tars = torch.cat(tars, dim=0)

    # TensorDataset + DataLoader
    dataset = TensorDataset(inps, tars)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader



import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wikitext2_testloader(nsamples, seed, seqlen, model, tokenizer=None, batch_size=1):
    """
    WikiText-2 test split から、max_length以内の分割済みトークン列をDataLoaderで返す。
    """
    # データセット読み込み
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # トークナイザ
    if tokenizer is None:
        print("Loading tokenizer...", model)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # テキストを結合してトークナイズ
    print("Tokenizing test data...")
    encodings = tokenizer("\n\n".join(testdata['text']), return_tensors='pt', add_special_tokens=False)

    input_ids = encodings.input_ids[0]

    # シーケンス長ごとに分割
    num_chunks = len(input_ids) // seqlen
    input_ids = input_ids[:num_chunks * seqlen]  # 端数を切り捨て
    input_ids = input_ids.view(num_chunks, seqlen)

    # nsamples分をランダムに抽出 (Noneの場合は全てのテストデータを使用)
    if nsamples is not None and nsamples < num_chunks:
        torch.manual_seed(seed)
        indices = torch.randperm(num_chunks)[:nsamples]
        input_ids = input_ids[indices]

    # DataLoader作成
    dataset = TensorDataset(input_ids, input_ids)  # 入力=出力（次トークン予測）
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return test_loader




@torch.no_grad()
def compute_ppl_from_testloader(model, testloader, device="cuda"):
    print("Evaluating ...")
    model.eval()
    model.to(device)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for inp, tar in tqdm(testloader, desc="Computing PPL"):
        inp, tar = inp.to(device), tar.to(device)
        outputs = model(inp)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tar[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        valid_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Perplexity: {ppl.item():.4f}")
    return ppl.item()


@torch.no_grad()
def compute_ppl_from_testenc(model, testenc, seqlen, device="cuda"):
    print('Evaluating ...')
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model = model.to(device)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        def __getattr__(self, name):
            # module に存在する属性は module から取ってくる
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)
    
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        # batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    for i in range(len(layers)):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        inps, outs = outs, inps
    
    model.to(device)
    testenc = testenc.to(device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print('perplexity:', ppl.item())
    model.config.use_cache = use_cache

    return ppl.item()


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer=None):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    if tokenizer is None:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    if tokenizer is None:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer=None):
    from datasets import load_dataset
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )

    # valdata = load_dataset("allenai/c4", "en", split="validation[:1%]")


    if tokenizer is None:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model, tokenizer=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model, tokenizer=None):
    from datasets import load_dataset
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )

    traindata = load_dataset(
        'allenai/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    # valdata = load_dataset(
    #     'allenai/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    valdata = load_dataset("allenai/c4", "en", split="validation[:1%]")


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', tokenizer=None
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, tokenizer)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, tokenizer)

        return get_c4(nsamples, seed, seqlen, model)