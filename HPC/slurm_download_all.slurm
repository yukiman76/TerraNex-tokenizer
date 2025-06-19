#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH --output=logs/hf_download_%j.out
#SBATCH --error=logs/hf_download_%j.err
#SBATCH --time=10-00:00:00                # Up to 10 days for large datasets
#SBATCH --mem=128G                        # Optimized for  downloader (was 512G)
#SBATCH --cpus-per-task=8                 # Balanced for concurrent downloads
#SBATCH --nice=0
#SBATCH --requeue                         # Allow automatic restart on node failure
#SBATCH --signal=USR1@120                 # Graceful shutdown 2min before kill (was 90s)
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=sonnym@hotmail.se

# 🏢 HuggingFace Dataset Downloader - Production Slurm Job


set -euo pipefail
export PYTHONUNBUFFERED=1
function cleanup_and_exit() {
    echo "🔄 [$(date)] Received shutdown signal - performing graceful cleanup..."
    python -c "import hf_downloader; hf_downloader.cleanup_locks()" 2>/dev/null || true
    echo "✅ [$(date)] Cleanup completed"
    exit 0
}

function health_check() {
    echo "🏥 [$(date)] Running  health checks..."
    python -c "
import hf_downloader
import sys

print('📋 System Health Report:')
print(f'   Configuration: {\"✅\" if hf_downloader.validate_configuration() else \"❌\"}')
print(f'   Network: {\"✅\" if hf_downloader.check_network_connectivity() else \"❌\"}')
print(f'   Memory: {hf_downloader.check_memory_usage():.1f}%')
print(f'   Cache: {\"✅\" if hf_downloader.check_cache_permissions() else \"❌\"}')
print(f'   Disk Space: {\"✅\" if hf_downloader.check_disk_space(100.0) else \"❌\"}')

# Exit with error if any critical checks fail
if not all([
    hf_downloader.validate_configuration(),
    hf_downloader.check_network_connectivity(),
    hf_downloader.check_cache_permissions(),
    hf_downloader.check_disk_space(100.0)
]):
    print('❌ Critical health checks failed')
    sys.exit(1)
else:
    print('✅ All health checks passed - ready for production')
"
}


trap cleanup_and_exit USR1 TERM INT

echo "🏢 =====  HF DATASET DOWNLOADER ====="
echo "🚀 Starting production dataset download pipeline"
echo "📅 Job started: $(date)"
echo "🧠 Compute node: $(hostname)"
echo "🔒 SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "📊 Resources: ${SLURM_MEM_PER_NODE}MB RAM, ${SLURM_CPUS_PER_TASK} CPUs"
echo "⏰ Time limit: ${SLURM_TIME_LIMIT}"
echo "📧 Notifications: ${SLURM_MAIL_USER}"
echo ""
echo "🔧 Setting up  environment..."


if command -v module &> /dev/null; then
    module load anaconda 2>/dev/null || echo "⚠️  Warning: anaconda module not found"
fi


# 🔑 HuggingFace Token Configuration
# Option 1: Use token from environment (recommended)
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "⚠️  Warning: HF_TOKEN not set - some datasets may require authentication"
    echo "💡 Set HF_TOKEN environment variable or run 'huggingface-cli login'"
else
    echo "✅ HF_TOKEN configured"
    export HF_TOKEN="$HF_TOKEN"
fi

# Option 2: Set token directly in script
export HF_TOKEN="hf_your_token_here"  # 🔄 REPLACE WITH YOUR ACTUAL TOKEN

# Optional: Set custom cache directory
# export HF_HOME="/path/to/your/cache"

# Activate environment (update this to your actual environment name)
ENV_NAME="tokenizer"  # 🔄 UPDATE THIS TO YOUR ACTUAL ENVIRONMENT NAME
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "📦 Current conda environment: $CONDA_DEFAULT_ENV"
else
    echo "📦 Activating conda environment: $ENV_NAME"
    source activate "$ENV_NAME" || {
        echo "❌ Failed to activate environment '$ENV_NAME'"
        echo "💡 Please update ENV_NAME in this script to your actual environment"
        exit 1
    }
fi


echo "🐍 Python version: $(python --version)"
echo "📍 Python location: $(which python)"


echo ""
echo "📁 Setting up  directory structure..."
mkdir -p logs datasets datasets/_locks datasets/_status


chmod 755 logs datasets datasets/_locks datasets/_status 2>/dev/null || true


echo ""
health_check


echo ""
echo "⬇️  Starting  download pipeline..."
echo "🔒 Security: Whitelist validation enabled"
echo "🛡️  Reliability: Exponential backoff with jitter"
echo "📊 Monitoring: Real-time resource tracking"
echo "🔄 Recovery: Automatic retry with lock management"


echo ""
echo "📥 Phase 1:  Dataset Download"
python hf_downloader.py --download-only --force

DOWNLOAD_EXIT_CODE=$?
echo "📊 Download phase exit code: $DOWNLOAD_EXIT_CODE"


if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🔍 Phase 2:  Dataset Verification"
    python hf_downloader.py --verify
    VERIFY_EXIT_CODE=$?
    echo "📊 Verification phase exit code: $VERIFY_EXIT_CODE"
else
    echo "⚠️  Skipping verification due to download errors"
    VERIFY_EXIT_CODE=1
fi


echo ""
echo "🧹 Phase 3:  Cleanup"
python hf_downloader.py --cleanup


echo ""
echo "📈 ===== COMPLETION REPORT ====="
echo "📅 Job completed: $(date)"
echo "⏱️  Total runtime: $SECONDS seconds"


python -c "
import hf_downloader
import json

try:
    status = hf_downloader.load_status()
    total = len(status)
    done = sum(1 for v in status.values() if v == 'done')
    failed = sum(1 for v in status.values() if v == 'failed')
    corrupt = sum(1 for v in status.values() if v == 'corrupt')
    
    print(f'📊 Dataset Status Summary:')
    print(f'   Total datasets: {total}')
    print(f'   ✅ Successfully downloaded: {done}')
    print(f'   ❌ Failed downloads: {failed}')
    print(f'   🔧 Corrupt/needs repair: {corrupt}')
    print(f'   📈 Success rate: {(done/total*100):.1f}%' if total > 0 else '   📈 Success rate: N/A')
    
    if failed > 0 or corrupt > 0:
        print(f'⚠️  Issues detected - review logs for details')
        
except Exception as e:
    print(f'❌ Error generating status report: {e}')
"


if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
    echo "🎉  DOWNLOAD PIPELINE COMPLETED SUCCESSFULLY"
    echo "✅ All datasets processed with -grade reliability"
    exit 0
else
    echo "⚠️   DOWNLOAD PIPELINE COMPLETED WITH ISSUES"
    echo "📋 Check logs for detailed error analysis"
    echo "🔄 Job may be automatically requeued by Slurm"
    exit $DOWNLOAD_EXIT_CODE
fi
