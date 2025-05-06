# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.color import rgb2lab
from tqdm import tqdm

def invert_jet_colormap_lab_3features(image_path, output_path=None):
    """
    jetカラーマップの逆変換を行い、3種類の特徴量
    - R: 元のスカラー
    - G: 小さい値をlog強調
    - B: 大きい値を逆log強調
    として3チャンネル画像を保存する関数。

    Args:
        image_path (str): 入力画像のファイルパス
        output_path (str, optional): 出力画像の保存先パス。指定しなければ、元ファイル名に .png を付けて保存する
    """
    # jetカラーマップを準備
    cmap = plt.cm.jet
    N = 256
    color_table = cmap(np.linspace(0, 1, N))[:, :3]  # RGB(0-1)
    color_table_lab = rgb2lab(color_table.reshape(1, N, 3)).reshape(N, 3)  # (256,3)

    # 入力画像を読み込む
    img = Image.open(image_path).convert('RGB')
    img_np = np.asarray(img) / 255.0  # (h,w,3), 0-1に正規化
    h, w, _ = img_np.shape

    # 入力画像もLabに変換
    img_lab = rgb2lab(img_np)
    
    # --- ベクトル化した処理 ---
    # 画像の形状を変更して処理 (h,w,3) -> (h*w,3)
    img_lab_flat = img_lab.reshape(-1, 3)
    
    # 一度に全ピクセルの距離計算を行う
    # 各ピクセルとカラーテーブルの各色との間の距離を計算
    # 結果は (h*w, N) の行列になる
    dists = np.zeros((img_lab_flat.shape[0], N), dtype=np.float32)
    
    # メモリ効率のため、バッチ処理を行う
    batch_size = 10000  # バッチサイズを調整して最適化
    n_batches = (img_lab_flat.shape[0] + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, img_lab_flat.shape[0])
        batch = img_lab_flat[start_idx:end_idx]
        
        # バッチごとに距離計算
        for j in range(N):
            dists[start_idx:end_idx, j] = np.linalg.norm(batch - color_table_lab[j], axis=1)
    
    # 各ピクセルの最小距離のインデックスを取得
    best_indices = np.argmin(dists, axis=1)
    
    # スカラー値に変換して元の形状に戻す
    scalars_img = (best_indices / (N - 1)).reshape(h, w)
    
    # --- 3つの特徴量を作る ---
    # R: 元のスカラー値
    scalar_r = scalars_img * 250
    img_3ch = np.stack([
        scalar_r.astype(np.uint8),
        scalar_r.astype(np.uint8),
        scalar_r.astype(np.uint8)
    ], axis=-1)  # (h, w, 3)
    
    # 保存
    out_img = Image.fromarray(img_3ch, mode='RGB')
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}.png"
    out_img.save(output_path)


# ../dataset/image の画像に対してすべてこの操作を実行し、 ../dataset/image_reverse に保存する
if __name__ == "__main__":
    input_dir = "../dataset/images"
    output_dir = "../dataset/images_reverse"
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            invert_jet_colormap_lab_3features(input_path, output_path)


