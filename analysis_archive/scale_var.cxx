
/* scale_weight tfile ttree factor */
int main(int argc, char *argv[])
{
  if (argc != 4)
    return 1;

  TFile ifile(argv[1]);
  TTree *tree = (TTree*)ifile.Get(argv[2]);
  double factor = std::strtod(argv[3]);

  TFile ofile(std::string(argv[1]) + ".reweighted");
  new_tree = tree.CloneTree(0);

  for (Long64_t i = 0; i < tree->GetEntries(); i++) {
    tree->Get
  
   
