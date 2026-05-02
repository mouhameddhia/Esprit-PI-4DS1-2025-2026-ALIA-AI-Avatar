param(
    [string]$PythonExe = "c:/Users/SBS/Desktop/ALIA/Multi-agents-Alia/knowledge_retrieval_agent/.venv/Scripts/python.exe"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $projectRoot

try {
    $judgeScores = @{
        overall_score = 0.82
        faithfulness = 0.90
        answer_relevance = 0.80
        context_utilization = 0.85
        medical_safety = 0.76
        clarity = 0.80
        mode_alignment = 0.82
    }

    $judgeScoresFile = Join-Path $projectRoot "scripts\sample_judge_scores.json"
    $judgeScores | ConvertTo-Json -Compress | Set-Content -Path $judgeScoresFile -Encoding UTF8

    & $PythonExe -m app.main xai-report `
        --question "What is the dosage of ferbiotic?" `
        --answer "Dosage is 10 ml." `
        --context "[FERBIOTIC#page=3] Dosage: 10 ml. Warning: avoid use in pregnancy." `
        --crag-decision CORRECT `
        --judge-scores-file $judgeScoresFile
}
finally {
    Pop-Location
}
