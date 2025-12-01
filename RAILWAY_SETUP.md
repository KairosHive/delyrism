# Railway Persistent Volume Setup

To avoid downloading models every time your app restarts, follow these steps in your Railway Dashboard:

## 1. Create a Volume
1. Go to your **delyrism** service in Railway.
2. Click on the **Volumes** tab (or "Storage").
3. Click **New Volume**.
4. Mount path: `/app/cache` (or any path you prefer).

## 2. Configure Environment Variables
Go to the **Variables** tab and add the following:

| Variable Name | Value | Description |
|--------------|-------|-------------|
| `HF_HOME` | `/app/cache/huggingface` | Tells Hugging Face to save models here. |
| `TORCH_HOME` | `/app/cache/torch` | Tells PyTorch to save data here. |
| `XDG_CACHE_HOME` | `/app/cache` | General cache directory. |

## 3. Redeploy
Once these are set, redeploy your application. 
- The first time you run a model, it will download to this volume.
- **Future restarts** will find the files in `/app/cache` and load them instantly without downloading.
