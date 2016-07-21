#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <TFile.h>
#include <TTree.h>

void split_tree(TTree *tree,
		 Long64_t start,
		 Long64_t nentries,
		 std::string& name)
{
	tree->GetEntry(0);
	TFile outfile(name.c_str(), "CREATE");
	TTree *new_tree = tree->CloneTree(0);

	Long64_t stop = start + nentries;
	assert(stop < tree->GetEntries());
	for (Long64_t i = start; i < stop; i++) {
		tree->GetEntry(i);
		new_tree->Fill();
	}

        outfile.Write(0, TObject::kWriteDelete);
}


// usage: split_tree <input> <ntrain> <nvalid> <ntest> <output_base>
//        ^0         ^1      ^2       ^3       ^4      ^5
int main(int argc, char *argv[])
{
	if (argc != 6) {
		// FIXME error message
		return 1;
	}

	TFile input(argv[1], "READ");
	if (input.IsZombie()) {
		std::fprintf(stderr, "unable to open input file: %s\n", argv[1]);
		return 1;
	}

	TTree *tree = (TTree*)input.Get("NNinput");
	if (tree == NULL) {
		std::fprintf(stderr, "unable get tree: NNinput from file: %s\n",
			     argv[1]);
		return 1;
	}

	Long64_t ntrain = std::atol(argv[2]);
	Long64_t nvalid = std::atol(argv[3]);
	Long64_t ntest = std::atol(argv[4]);

	std::string output_base(argv[5]);
	std::string output_train = output_base + ".training.root";
	std::string output_valid = output_base + ".validation.root";
	std::string output_test = output_base + ".test.root";

	split_tree(tree, 0, ntrain, output_train);
	split_tree(tree, ntrain, nvalid, output_valid);
	split_tree(tree, ntrain + nvalid, ntest, output_test);

	return 0;
}
