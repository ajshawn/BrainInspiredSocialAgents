import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# -------------------------------------------------------------------------
# Helper to find the first index where array >= threshold
# Returns None if no such index is found
# -------------------------------------------------------------------------
def first_intersection(arr, val):
  idx = np.where(arr >= val)[0]
  return idx[0]+1 if idx.size > 0 else None  # +1 because python is 0-based

# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------
def plot_ve_results_python():
  file_names = [
    './variance_exp_results/compiledMat_concat_AH.mat',
    './variance_exp_results/compiledMat_concat_cp7357.mat',
    './variance_exp_results/compiledMat_concat_cp9651.mat',
  ]
  result_names = ['AH', 'open7357', 'open9651']

  # Make sure output directory exists
  out_dir = './variance_exp_results_python/'
  os.makedirs(out_dir, exist_ok=True)

  for file_id, file_name in enumerate(file_names):
    result_name = result_names[file_id]

    print(f'Loading {file_name}')
    data = sio.loadmat(file_name, squeeze_me=True, struct_as_record=False)
    # 'compiledMat' may be a 1x1 object array in the .mat file
    compiledMat = data['compiledMat']

    # Extract .plsc and .randNull25 from compiledMat
    # Depending on how MATLAB saved them, you may need to adjust indexing slightly
    plsc       = compiledMat.plsc     # shape: (nPred, nPrey, 2)
    randNull25 = compiledMat.randNull25  # shape: (nPred, nPrey), each cell is 25x2 or None

    # Get dimensions
    nPred, nPrey, _ = plsc.shape

    PCcountPred = np.full((nPred, nPrey), np.nan)  # #PC needed for predator
    PCcountPrey = np.full((nPred, nPrey), np.nan)  # #PC needed for prey

    # 1) Plot subplots of Rand-25-PC curves vs. PLSC lines
    fig1, axs1 = plt.subplots(
      nrows = nPred,
      ncols = max(nPrey - nPred, 1),  # to avoid zero or negative
      figsize=(14, 8),
      squeeze=False
    )
    fig1.canvas.manager.set_window_title(f'Rand25 vs PLSC: {result_name}')
    # Attempt to maximize the figure window
    try:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()
    except:
      pass

    subInd = 0
    for p in range(nPred):
      for r in range(nPred, nPrey):
        veRand = randNull25[p][r]   # should be a 25x2 array or None
        if veRand is None or veRand.size == 0 or np.all(np.isnan(veRand)):
          continue
        # plsc[p,r,0] => predator; plsc[p,r,1] => prey
        plscPred = plsc[p, r, 0]
        plscPrey = plsc[p, r, 1]

        ax = axs1[p, r - nPred]  # we shift columns by (r-nPred)
        ax.set_title(f'p={p}, r={r}')

        # Plot the random-PC curves
        xvals = np.arange(1, veRand.shape[0] + 1)
        ax.plot(xvals, veRand[:,0], 'r-', linewidth=2, label='Pred (rand25)')
        ax.plot(xvals, veRand[:,1], 'b-', linewidth=2, label='Prey (rand25)')

        # Dashed horizontal lines for PLSC measure
        ax.axhline(plscPred, color='r', linestyle='--', linewidth=1.5)
        ax.axhline(plscPrey, color='b', linestyle='--', linewidth=1.5)

        # Find intersection
        idxPred = first_intersection(veRand[:,0], plscPred)
        idxPrey = first_intersection(veRand[:,1], plscPrey)
        if idxPred is not None:
          PCcountPred[p, r] = idxPred
          ax.plot(idxPred, plscPred, 'ro', markersize=5)
        if idxPrey is not None:
          PCcountPrey[p, r] = idxPrey
          ax.plot(idxPrey, plscPrey, 'bo', markersize=5)

        ax.set_xlim([1, 25])
        ax.set_xlabel('#PC')
        ax.set_ylabel('VE difference (obs - perm)')

        # Label with "pred #X PCs; prey #Y PCs"
        lbl_pred = str(idxPred) if idxPred else "None"
        lbl_prey = str(idxPrey) if idxPrey else "None"
        ax.set_title(f'p={p},r={r} | pred #{lbl_pred} PCs; prey #{lbl_prey} PCs')

    fig1.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1.15,1.15))
    out_fig1 = os.path.join(out_dir, f'{result_name}_randPCs.png')
    plt.savefig(out_fig1, dpi=150)
    print(f'Saved {out_fig1}')

    # 2) Heatmaps for predator & prey
    # Predator
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.canvas.manager.set_window_title(f'Heatmap #PC (predator) {result_name}')
    try:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()
    except:
      pass
    im2 = ax2.imshow(PCcountPred, aspect='auto', origin='upper')
    fig2.colorbar(im2, ax=ax2)
    ax2.set_title(f'#PC needed (predator) - {result_name}')
    ax2.set_xlabel('Prey index')
    ax2.set_ylabel('Predator index')
    out_fig2 = os.path.join(out_dir, f'{result_name}_heatmap_pred.png')
    plt.savefig(out_fig2, dpi=150)
    print(f'Saved {out_fig2}')

    # Prey
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    fig3.canvas.manager.set_window_title(f'Heatmap #PC (prey) {result_name}')
    try:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()
    except:
      pass
    im3 = ax3.imshow(PCcountPrey, aspect='auto', origin='upper')
    fig3.colorbar(im3, ax=ax3)
    ax3.set_title(f'#PC needed (prey) - {result_name}')
    ax3.set_xlabel('Prey index')
    ax3.set_ylabel('Predator index')
    out_fig3 = os.path.join(out_dir, f'{result_name}_heatmap_prey.png')
    plt.savefig(out_fig3, dpi=150)
    print(f'Saved {out_fig3}')

    # 3) Histogram of #PC required
    allPredCounts = PCcountPred[~np.isnan(PCcountPred)]
    allPreyCounts = PCcountPrey[~np.isnan(PCcountPrey)]

    fig4, ax4 = plt.subplots(figsize=(8,5))
    fig4.canvas.manager.set_window_title(f'Histogram #PC required {result_name}')
    try:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()
    except:
      pass

    bins = np.arange(1, 26)  # 1..25
    ax4.hist(allPredCounts, bins=bins, alpha=0.5, label='Predator')
    ax4.hist(allPreyCounts, bins=bins, alpha=0.5, label='Prey')
    ax4.set_xlabel('#PCs')
    ax4.set_ylabel('Counts')
    ax4.legend(loc='best')
    ax4.set_title(f'Distribution of #PCs required - {result_name}')
    out_fig4 = os.path.join(out_dir, f'{result_name}_hist.png')
    plt.savefig(out_fig4, dpi=150)
    print(f'Saved {out_fig4}')

    # (Optional) show or close
    # plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

  print('All done!')

if __name__ == '__main__':
  plot_ve_results_python()
