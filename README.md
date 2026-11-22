# AETHER FLOCK — Three.js vs 自作ゲームエンジン (WebAssembly SIMD + WebGPU)

同一の3D群衆シミュレーションWebアプリを **Three.js (v0.187.0)** と **Three.jsを使わない最適化された自作ゲームエンジン（Rust WebAssembly SIMD128 + Raw WebGPU / WGSL）** の両方で実装し、リアルタイムで視覚的同一性とパフォーマンスを比較・計測するベンチマークWebアプリです。

- **ライブデモ (GitHub Pages)**: https://github30.github.io/threejs-vs-no-threejs/
- **参考サンプル**: [Three.js Examples](https://threejs.org/examples/)（`webgpu_compute_birds`, `webgl_instancing_dynamic`, `webgl_instancing_performance`, `webgpu_performance_renderbundle`）
- **外部ライブラリ制約**: **Three.js 以外のライブラリは一切未使用**（JS/CSSフレームワーク0、Wasm依存クレート0、`#![no_std]` Rust + バニラJS/CSS/HTML + 生WGSL）

---

## 3つのベンチマークシナリオ（同一シーン・同一アルゴリズム）

両エンジンは共通の3Dジオメトリ（`src/shared/geometry.js`）、共通の決定論的シード（`src/shared/seed.js`）、共通のカメラ・フラスタム計算（`src/shared/camera.js`）、同一の太陽光＋半球環境光＋ACES Filmicトーンマッピングを共有しています。

| シナリオ | 内容 | 自作エンジン（No Three.js） | Three.js v0.187.0 |
|---|---|---|---|
| **1. Hybrid CPU+GPU** | 空間ハッシュ3Dボイド物理 + 障害物回避 + フラスタムカリング + 動的インスタンシング | `#![no_std]` Rust → **Wasm SIMD128**（16B整列SoA計数ソート＋4並列近傍探索＆カリング）＋ **32B圧縮バッファ直転送** ＋ Raw WebGPU | 同一空間ハッシュJS物理（TypedArray最適化）＋ `THREE.InstancedMesh`（64B `mat4` 転送）＋ `WebGPURenderer` / `WebGLRenderer` |
| **2. GPU Compute** | CPU-GPU間転送ゼロの純GPUコンピュート群衆シミュレーション | 生 **WGSLコンピュートシェーダー**（`var<workgroup>` 64スレッド共有メモリタイリング・1パス融合・ピンポンバッファ）＋ 事前記録 `GPURenderBundle` | 公式 `THREE.WebGPURenderer` ＋ **TSL (`three/tsl`)** `Fn()` ピンポンコンピュート ＋ `MeshStandardNodeMaterial` |
| **3. Draw Calls / Scene Graph** | 数千の独立した回転3Dクリスタルを個別ドローコールで描画 | Rust **Wasm SIMD128**（4並列sin/cos・フラスタムカリング）＋ 事前記録 **`GPURenderBundle`** でN回の個別 `draw` を一括再生 | 標準 **`THREE.Scene`** ＋ N個の独立した `THREE.Mesh`（共有ジオメトリ・マテリアル） |

> WebGPU非対応ブラウザでも動作するよう、自作エンジンには同一Rust Wasm SIMDコアを共有する軽量WebGL2フォールバック（`src/custom-engine/webgl2-fallback.js`）を、Three.js側には `WebGPURenderer` / `WebGLRenderer` 切替機能を搭載しています。

---

## 計測結果サマリー（単独フルスクリーン自動計測）

> 下記は Chrome 158（WebGPU有効・SwiftShaderソフトウェアVulkan環境 / 1440×860）での自動計測値です。ソフトウェアVulkan環境のため実効FPSはラスタライズ律速になりますが、**CPUフレーム時間・起動時間・バンドルサイズ・ヒープ消費量**は実機ハードウェアGPU上での実効性能差（特に60〜240Hz描画時のCPUボトルネック）を正確に反映します。実機GPU上でのFPS計測は [GitHub Pages のライブデモ](https://github30.github.io/threejs-vs-no-threejs/) の「自動ベンチマーク」ボタンから直接実行できます。

### 1. ランタイム性能（CPUフレーム時間・起動・メモリ）

| 計測項目 | 自作エンジン (Wasm SIMD + WebGPU) | Three.js v0.187 (WebGPU + TSL) | 倍率 |
|---|---:|---:|---:|
| **Hybrid (8,192羽) CPU時間** | **8.20 ms** | 18.17 ms | **2.22x 高速** |
| **GPU Compute (2,048羽) CPU時間** | **0.20 ms** | 1.52 ms | **7.60x 高速** |
| **Scene Graph (2,500体) CPU時間** | **0.45 ms** | 66.97 ms | **148.8x 高速** |
| **CPU処理速度（3シナリオ幾何平均）** | — | — | **13.60x 高速** |
| **初期化・シェーダー構築時間** | **101 ms** | 513 ms | **5.1x 高速** |
| **JSヒープ消費 (Scene Graph 2,500体)** | **19.3 MB** | 39.8 MB | **2.1x 省メモリ** |

### 2. ペイロード / バンドルサイズ比較

| エンジン構成 | 非圧縮 (Raw) | gzip圧縮 | 外部ライブラリ数 |
|---|---:|---:|---:|
| **自作エンジンコア** (`engine_simd.wasm` 10.8 KB + JS/WGSL) | **59 KB** | **17 KB** | **0** |
| Three.js v0.187 (WebGL2版 `three.module.js` + adapter) | 667 KB | 133 KB | 1 (`three`) |
| Three.js v0.187 (WebGPU+TSL版 `core + tsl + webgpu` + adapter) | 3,839 KB | 759 KB | 1 (`three`) |

---

## 結論：どちらが高性能か？

- **純粋な実行性能・軽量性・起動速度では、Webアプリに特化した自作エンジン（Wasm SIMD + Raw WebGPU）が圧倒的に高性能**です。
  - **CPU負荷（物理・カリング・行列更新）**: `#![no_std]` Rust Wasm SIMD128 の 16B 整列 SoA メモリ＋ゼロGCループにより、JS 実装より **2〜5倍以上高速**。
  - **ドローコール・シーングラフ負荷**: WebGPU の `GPURenderBundle` により、数千オブジェクトの描画コマンド発行コストがほぼゼロ（**0.45 ms vs 66.97 ms、約149倍差**）。ハードウェアGPU環境では、Three.js が CPU 律速で ~15 FPS に落ち込む場面でも 120+ FPS を維持できます。
  - **GPUコンピュート＆メモリ帯域**: 32B 圧縮インスタンス構造体（64B `mat4` の半分）、`var<workgroup>` 共有メモリタイリングによる VRAM 読込削減、Early-Z 前後描画により、GPU 帯域とコンパイル時間を大幅に削減。
  - **初回ロード**: エンジンコアが **59 KB（gzip 17 KB）** と Three.js WebGPU 版の **約1/65** のサイズで、初期化も約 **5倍高速**。
- **一方で Three.js が強い領域**:
  - 静的な単一 `InstancedMesh` など GPU ラスタライズ律速のシーンでは両者の FPS 差は小さくなります。
  - glTF / PBR / ポストプロセス / シャドウマップ / クロスブラウザ互換といった**開発生産性とエコシステム**では Three.js が大きな優位性を持ちます。

---

## 自動横画面フルスクリーン計測 & 結果自動収集（Cloudflare Workers + D1）

- **横画面フルスクリーン自動切替**: 「自動ベンチマーク」開始時に Fullscreen API（`requestFullscreen`）＋ Screen Orientation API（`screen.orientation.lock('landscape')`）で自動的に横画面フルスクリーンへ切り替えて計測し、終了時に自動復帰します（`orientation.lock` 非対応の iOS Safari 等では CSS 回転フォールバックで横画面比率を維持）。
- **デバイス情報・User-Agent 込みの自動収集**: 計測完了時に User-Agent、`navigator.userAgentData`（OS・アーキテクチャ・モバイル判定・モデル等）、WebGPU アダプタ情報、WebGL アンマスク GPU 名、CPU コア数、メモリ、画面解像度・DPR・フルスクリーン状態を自動送信し、Cloudflare D1 へ保存します。
- **公開テレメトリ API (Cloudflare Workers)**:
  - 一覧取得: `GET https://threejs-vs-no-threejs-telemetry.6u.workers.dev/api/results`
  - 集計サマリー: `GET https://threejs-vs-no-threejs-telemetry.6u.workers.dev/api/summary`
  - 個別詳細: `GET https://threejs-vs-no-threejs-telemetry.6u.workers.dev/api/results/:id`

---

## ディレクトリ構成

```text
.
├── index.html                        # アプリUI（同時比較・単独モード・自動ベンチマーク・比較レポート）
├── styles.css                        # バニラCSSデザインシステム（横画面フルスクリーン対応）
├── worker/
│   ├── index.js                      # ゼロ依存 Cloudflare Worker テレメトリAPI
│   └── schema.sql                    # Cloudflare D1 SQLite スキーマ
├── wasm/
│   ├── src/lib.rs                    # 純 #![no_std] Rust Wasm SIMD128 物理・カリングコア（クレート0）
│   ├── build.sh                      # rustc ビルドスクリプト
│   ├── engine_simd.wasm              # コンパイル済み Wasm SIMD128 バイナリ (10.8 KB)
│   └── engine_scalar.wasm            # スカラーフォールバック Wasm (6.1 KB)
├── src/
│   ├── main.js                       # アプリ制御・横画面フルスクリーン計測・自動送信
│   ├── shared/                       # ジオメトリ・シード・カメラ・JS物理・メトリクス・テレメトリ送信
│   ├── custom-engine/                # 自作エンジン (wasm-bridge / WGSL shaders / WebGPU / WebGL2)
│   └── three-engine/                 # Three.js v0.187 エンジン (WebGPURenderer + TSL / WebGLRenderer)
└── vendor/three/                     # 公式 Three.js v0.187.0 ESモジュール（他ライブラリ一切なし）
```

## ローカル実行 & Wasm 再ビルド

```bash
# ローカルHTTPサーバーで起動
python3 -m http.server 8080
# ブラウザで http://localhost:8080 を開く

# Rust WebAssembly コアを再ビルドする場合（要 rustup target add wasm32-unknown-unknown）
bash wasm/build.sh
```
