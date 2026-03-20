import os
from pathlib import Path


# バイナリファイルからロード
def fvecs_read(file_name):
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        d = data.view(np.int32)[0]  # 次元数
        return data.reshape(-1, d + 1)[:, 1:]


def get_matrix_data():
    torch.manual_seed(1)
    # 実験用行列
    w1 = torch.randn(128, 128)
    torch.save(w1, os.path.dirname(__file__)+'/data/matrix_data/random_matrix.pt')
    print('Random data', w1.shape)

    # DeiT-S重み
    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    model_weights = model.state_dict()
    w_name = ['intermediate', 'key', 'value', 'dense', 'query']
    for i, (name, param) in enumerate(model_weights.items()):
        w2 = param.data
        if i==8:
            break
    torch.save(w2, os.path.dirname(__file__)+'/data/matrix_data/deit-s_matrix.pt')
    print('DeiT-S data', w2.shape)


    # TSPLIB距離グラフ
    problem_name = 'rd100'
    problem_root = Path(os.getenv("TSPLIB_DIR", Path(__file__).resolve().parents[3] / "problem" / "tsp"))
    problem_path = problem_root / problem_name / f"{problem_name}.tsp"
    instance = TSP(problem_path)
    w3 = torch.tensor(instance.Dij)
    torch.save(w3, os.path.dirname(__file__)+'/data/matrix_data/rd100_distance_matrix.pt')
    print('TSP data', w3.shape)

    # ANNデータ
    dataname_list = ['siftsmall', 'sift']
    data_name = dataname_list[1]
    ann_root = Path(os.getenv("ANN_DATASET_DIR", Path(__file__).resolve().parents[3] / "ann" / "dataset"))
    w4 = torch.tensor(fvecs_read(str(ann_root / data_name / f"{data_name}_base.fvecs"))[:128])
    torch.save(w4, os.path.dirname(__file__)+'/data/matrix_data/sift_matrix.pt')
    print('ANN data', w4.shape)


    # ImageNet
    data_path = os.getenv("IMAGENET_DIR") or os.getenv("IMAGENET_ROOT") or os.getenv("ILSVRC2012_DIR")
    if data_path is None:
        raise ValueError("ImageNet path is not set. Set IMAGENET_DIR, IMAGENET_ROOT, or ILSVRC2012_DIR.")
    train, val = get_imagenet(model_name='deit', num_traindatas=32, data_path=data_path)
    for batch_idx, (images, _) in enumerate(train):
        w5 = images[5][0]
        break
    print('ImageNet data', w5.shape)
    torch.save(w5, os.path.dirname(__file__)+'/data/matrix_data/imagenet_matrix.pt')
    return [w1, w2, w3, w4, w5]
