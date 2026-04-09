$ErrorActionPreference = "Stop"

Write-Host "Installing git-xet (optional but useful for Hugging Face repos)..."
winget install git-xet

Write-Host "Installing Hugging Face CLI..."
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

Write-Host "Downloading StartZer0/az-sentiment-bert into task1_extended/models/az-sentiment-bert ..."
hf download StartZer0/az-sentiment-bert --local-dir task1_extended/models/az-sentiment-bert

Write-Host "Done."
