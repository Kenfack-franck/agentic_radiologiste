# UNBOXED Radiology — MCP Server

Serveur MCP exposant les outils du pipeline radiologique du hackathon UNBOXED.

## Tools disponibles

| Tool | Description |
|------|-------------|
| `list_patients` | Liste tous les patients Orthanc avec leurs infos registry |
| `get_patient_history(patient_id)` | Historique complet d'un patient depuis l'Excel |
| `run_segmentation(accession)` | Télécharge CT + segmentation → findings |
| `generate_report(accession)` | Pipeline complet → rapport radiologique markdown |
| `compare_exams(patient_id)` | Tableau d'évolution des lésions + RECIST |

## Configuration

### Prérequis

```bash
pip install pydicom SimpleITK requests pandas openpyxl google-genai mcp
pip install ./dcm_seg_nodules-1.0.0-py3-none-any.whl
```

### Clé API Gemini

Créer un fichier `work/.env` :
```
GEMINI_API_KEY=votre_clé_ici
```

Ou exporter dans l'environnement :
```bash
export GEMINI_API_KEY="votre_clé_ici"
```

## Connexion à Claude Desktop

Ajouter dans `~/.claude/claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "unboxed-radiology": {
      "command": "python3",
      "args": ["/home/kenfack/Documents/hackaton_lyon/work/mcp_server.py"],
      "env": {
        "GEMINI_API_KEY": "votre_clé_ici"
      }
    }
  }
}
```

## Connexion à Claude Code

Ajouter dans `.claude/settings.json` du projet :

```json
{
  "mcpServers": {
    "unboxed-radiology": {
      "command": "python3",
      "args": ["work/mcp_server.py"],
      "cwd": "/home/kenfack/Documents/hackaton_lyon",
      "env": {
        "GEMINI_API_KEY": "votre_clé_ici"
      }
    }
  }
}
```

## Connexion avec un client MCP générique (stdio)

```bash
python3 work/mcp_server.py
```

Le serveur communique via stdin/stdout au format JSON-RPC (protocole MCP).

## Test rapide

```python
# Depuis un script Python
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test():
    params = StdioServerParameters(
        command="python3",
        args=["work/mcp_server.py"],
        cwd="/home/kenfack/Documents/hackaton_lyon",
    )
    async with stdio_client(params) as (read, write):
        from mcp.client.session import ClientSession
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Tools: {[t.name for t in tools.tools]}")

            # List patients
            result = await session.call_tool("list_patients", {})
            print(result)

asyncio.run(test())
```

## Architecture

```
work/
├── pipeline.py          # Pipeline standalone (CLI)
├── mcp_server.py        # Serveur MCP (ce fichier)
├── 01_download_ct.py    # Script unitaire download
├── 02_segmentation.py   # Script unitaire segmentation
├── 03_generate_report.py# Script unitaire rapport
├── .env                 # Clé API Gemini
├── patients/            # Données patients (auto-créé)
│   ├── {accession}/
│   │   ├── original/    # Fichiers DICOM CT
│   │   ├── patient_summary.json
│   │   └── rapport_radiologique.md
│   └── _results/        # Fichiers SEG générés
└── README.md
```
