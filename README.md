# LLM Imagine Master

![](docs/QwenImageMaster.png)

Qwen Image Plugin for Dify

https://github.com/langgenius/dify-plugin-daemon/releases

```bash
uv pip compile pyproject.toml -o qwen-image-master/requirements.txt
```

```bash
mkdir -p difypkg
./dify-plugin-windows-amd64.exe plugin package qwen-image-master/ -o difypkg/qwen-image-master-0.0.3.difypkg
```