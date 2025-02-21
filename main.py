# データ作成に使用するライブラリ
from torchvision import datasets # データセット読み込み
import torchvision.transforms as transforms # 前処理用
from torch.utils.data import DataLoader # バッチ処理、シャッフル

# モデル作成に使用するライブラリ
import torch
import torch.nn as nn # ニューラルネット用のクラスやレイヤー
import torch.optim as optim # 最適化手法
import torch.nn.functional as F # reluなどの便利関数

# よく使用するライブラリ
import matplotlib.pyplot as plt # 描画
import numpy as np # PyTorchのテンソル操作

torch.manual_seed(1)

##################
###絶対先に定義###
##################

"""gpuの指定"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""ハイパーパラメータの定義"""
batch_size = 100 # バッチサイズ
n_channel = 100 # 生成器出力チャンネル数
n_epoch = 100 # 学習エポック数

"""データの読み込み"""

# trans.composeで複数処理を順番に適応、データセットの各画像に一貫した処理が可能
# ToTensor:# 画像をPyTrochのテンソル形式(torch.Tensor)に変換、ToTensorでピクセル値を[0, 255]から[0.0, 1.0]の範囲に正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

# True:訓練用60,000枚、False:テスト用10,000枚
# download:データセットがローカルすでにある場合ダウンロードしない
root = './data'
mnist_dataset = datasets.MNIST(
    root=root,download=True,
    train=True,
    transform=transform
    )

# PyTorchのDataLoderはデータセットをバッチ単位で読み込む
# trainset >> 上で作成した訓練用データをDataLoderに渡す
# バッチサイズ(1回の学習でモデルに渡すデータの数)を指定
# shuffle >> データを毎回シャッフルして学習を行う設定、学習の安定化、過学習防止
dataloader = DataLoader(
    mnist_dataset,
    batch_size=batch_size,
    shuffle=True
    )

"""生成器クラス"""

# Fully Conected（全結合層、FC層）を定義
# nn.Linear(input_dim, output_dim)：線形変換y=wx+bを行う
# view()：PyTorchのテンソル形状を変更する関数()
class Generator(nn.Module):
    # モデル初期化メソッド
    def __init__(self):
        # Generatorの__init__を呼び出す
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, 1, 28, 28)
        return x

"""判別機クラス"""

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu(self.fc1(x), negative_slope = 0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope = 0.2)
        x = torch.sigmoid(self.fc3(x))
        return x

  """インスタンス作成"""

# >> それぞれがメモリ上に準備され学習可能になる
# Generator(生成器)　　：ノイズから画像生成
# Discriminator(判別機)：その画像が本物か偽物か判断
# to(device)　　　　　：指定したデバイスにインスタンスを移動
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 損失関数、バイナリ交差エントロピ損失
criterion = nn.BCELoss()

# Adamのオプティマイザ設定
optimizerG = optim.Adam(generator.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999)
                        )
optimizerD = optim.Adam(discriminator.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999)
                        )

"""学習ループ"""
# １．データ読み込み、バッチ処理
# ２．判別機学習
#   ・本物画像で判別機を学習
#   ・偽物画像で判別機を学習
# ３．生成器の学習

G_losses = []
D_losses = []
D_x_list = []
D_G_z1_list = []
D_G_z2_list = []

# 学習のループ
for epoch in range(n_epoch):
    for i, (real_image, _) in enumerate(dataloader):

        # 前準備
        real_image = real_image.to(device) # ミニバッチをデバイスに転送

        real_target = torch.full((batch_size,), 1., device=device).unsqueeze(1) # 本物ラベル
        fake_target = torch.full((batch_size,), 0., device=device).unsqueeze(1) # 偽物ラベル

        noise = torch.randn(batch_size,n_channel,device=device) # ノイズ作成

        # discriminatorの学習(本物画像の学習)
        discriminator.zero_grad() # 勾配リセット
        output = discriminator(real_image) # 本物画像で判別
        errD_real = criterion(output, real_target) # 本物画像に対する損失
        D_x = output.mean().item() # .item()：テンソル形式の損失値を浮動小数点に変換するメソッド

        # discriminatorの学習(偽物画像の学習)
        fake_image = generator(noise) # ノイズから画像を生成
        output = discriminator(fake_image.detach()) # 生成した偽物を使って判別
        errD_fake = criterion(output, fake_target) # 偽物画像に対する損失
        D_G_z1 = output.mean().item() # 判別機が偽物を判別した確率

        # Dの損失を合計して逆伝番
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step() # 更新

        # generatorの学習
        generator.zero_grad() # 勾配リセット
        output = discriminator(fake_image) # 生成した偽物画像に対する判別結果
        errG = criterion(output, real_target) # 生成器に損失
        errG.backward()
        D_G_z2 = output.mean().item() # 判別機が生成器の偽物を判別した確率
        optimizerG.step()

    # 進捗の確認
    if (epoch + 1) % 10 == 0:

        print(f"Epoch [{epoch + 1}/{n_epoch}], Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")


        # 10回ごとに生成器でノイズから画像を生成し、偽画像を確認する
        noise = torch.randn(16, n_channel, device=device)  # ノイズベクトル
        fake_images = generator(noise)  # 生成器で偽画像を生成
        # 生成した画像を表示
        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        for i in range(16):  # 16枚の画像を表示
            ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
            ax.imshow(fake_images[i].cpu().detach().view(28, 28), cmap='gray')  # 画像を28x28に変形して表示
        plt.show()
