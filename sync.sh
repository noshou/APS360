  # Then run one rsync per subdir in parallel
  for dir in \
    "ar_ne_clusters" "QM9" "COD" "rcsb_big" "rcsb_med" \
    "rcsb_sml" "si_ge_clusters" "mofs" "tmQM" \
    "(NaCl)_nCl-" "(Na,Co,Ag,Pb,Mo,Fe)_monoatomic_clusters"; do
    rsync -aHAXxv --numeric-ids --no-compress \
      -e "ssh -T -o Compression=no -x" \
      --rsync-path=wsl \
      "data/xyz/${dir}/" \
      "nathan@192.168.2.162:/home/nathan/APS360/data/xyz/${dir}/" &
  done

  # Also sync the top-level files
  rsync -av --numeric-ids --no-compress \
    -e "ssh -T -o Compression=no -x" \
    --rsync-path=wsl \
    --include="*.json" --include="*.tsv" --include="*.txt" --include="*.sh" \
    --exclude="xyz/" --exclude="*/" \
    data/ nathan@192.168.2.162:/home/nathan/APS360/data/ &

  wait
  echo "All done"
