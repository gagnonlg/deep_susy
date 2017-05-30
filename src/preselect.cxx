#include <iostream>
#include <limits>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TVector2.h>

#define ERROR(msg) do {							\
	std::cerr << "ERROR " << msg << "\n"; std::exit(EXIT_FAILURE); \
	} while (0)

#define INFO(msg) do {				\
	std::cout << "INFO " << msg << std::endl; \
	} while (0)

void do_thinning(TChain &chain)
{
#define CONNECT(b)  do { \
		TObject* br = chain.GetListOfBranches()->FindObject(b); \
		if (br)							\
			chain.SetBranchStatus(b, 1);		\
		else							\
			ERROR("Branch \"" << b << "\" not found");	\
	} while (0)
	
	chain.SetBranchStatus("*", 0);
	CONNECT("event_number");
	CONNECT("run_number");
	CONNECT("muons_n");
	CONNECT("muons_pt");
	CONNECT("muons_phi");
	CONNECT("muons_eta");
	CONNECT("muons_e");
	CONNECT("electrons_n");
	CONNECT("electrons_pt");
	CONNECT("electrons_phi");
	CONNECT("electrons_eta");
	CONNECT("electrons_e");
	CONNECT("jets_pt");
	CONNECT("jets_phi");
	CONNECT("jets_eta");
	CONNECT("jets_e");
	CONNECT("jets_isb_60");
	CONNECT("jets_isb_70");
	CONNECT("jets_isb_77");
	CONNECT("jets_isb_85");
	CONNECT("mettst");
	CONNECT("mettst_phi");
	CONNECT("rc_R08PT10_jets_pt");
	CONNECT("rc_R08PT10_jets_phi");
	CONNECT("rc_R08PT10_jets_eta");
	CONNECT("rc_R08PT10_jets_e");
	CONNECT("rc_R08PT10_jets_m");
	CONNECT("rc_R08PT10_jets_nconst");
	CONNECT("pass_MET");
	CONNECT("weight_WZ_2_2");
	CONNECT("weight_btag");
	CONNECT("weight_elec");
	CONNECT("weight_jvt");
	CONNECT("weight_mc");
	CONNECT("weight_muon");
	CONNECT("weight_pu");
	CONNECT("weight_ttbar_NNLO");
	CONNECT("weight_ttbar_NNLO_1L");
	CONNECT("gen_filt_ht");
	CONNECT("gen_filt_met");
#undef CONNECT
}

void info_cutflow(TH1D *cutflow, bool ht_filter, bool met_filter)
{
	INFO("Cutflow: start: " << cutflow->GetBinContent(1));
	INFO("Cutflow: MET trigger: " << cutflow->GetBinContent(2));
	INFO("Cutflow: MET > 200: " << cutflow->GetBinContent(3));
	INFO("Cutflow: Njet > 4: " << cutflow->GetBinContent(4));
	INFO("Cutflow: Nb-jet > 2: " << cutflow->GetBinContent(5));
	INFO("Cutflow: dphi_min > 0.4: " << cutflow->GetBinContent(6));
	if (ht_filter)
		INFO("Cutflow: gen_filt_ht < 600: " << cutflow->GetBinContent(7));
	if (met_filter)
		INFO("Cutflow: gen_filt_met < 200: " << cutflow->GetBinContent(8));
}

TH1D* do_skimming(TChain &chain, TTree *tree, bool ht_filter, bool met_filter)
{
	Int_t pass_MET;
	chain.SetBranchAddress("pass_MET", &pass_MET);
	std::vector<Float_t> *jets_pt = nullptr;
	chain.SetBranchAddress("jets_pt", &jets_pt);
	std::vector<Float_t> *jets_eta = nullptr;
	chain.SetBranchAddress("jets_eta", &jets_eta);
	std::vector<Float_t> *jets_phi = nullptr;
	chain.SetBranchAddress("jets_phi", &jets_phi);
	std::vector<Int_t> *jets_isb_85 = nullptr;
	chain.SetBranchAddress("jets_isb_85", &jets_isb_85);
	Float_t mettst;
	chain.SetBranchAddress("mettst", &mettst);
	Float_t mettst_phi;
	chain.SetBranchAddress("mettst_phi", &mettst_phi);
	Int_t muons_n;
	chain.SetBranchAddress("muons_n", &muons_n);
	Int_t electrons_n;
	chain.SetBranchAddress("electrons_n", &electrons_n);
	Float_t gen_filt_ht;
	chain.SetBranchAddress("gen_filt_ht", &gen_filt_ht);
	Float_t gen_filt_met;
	chain.SetBranchAddress("gen_filt_met", &gen_filt_met);

	TH1D *h_cutflow = new TH1D("cutflow", "", 8, 0, 8);
	
	for (Long64_t i = 0; i < chain.GetEntries(); i++) {
		
		chain.GetEntry(i);
		h_cutflow->Fill(0);

		/* MET trigger & requirement */
		
		if (!pass_MET)
			continue;
		h_cutflow->Fill(1);

		if (mettst < 200)
			continue;
		h_cutflow->Fill(2);

		/* Jet requirement */
		
		int njets = 0;
		int nbjets = 0;
		for (size_t i = 0; i < jets_pt->size(); i++) {
			Float_t pt = jets_pt->at(i);
			Float_t eta = jets_eta->at(i);
			Int_t isb = jets_isb_85->at(i);
			if (pt > 30 && abs(eta) < 2.8)
				njets++;
			if (isb && pt > 30 && abs(eta) < 2.8)
				nbjets++;
		}

		if (njets < 4)
			continue;
		h_cutflow->Fill(3);

		if (nbjets < 2)
			continue;
		h_cutflow->Fill(4);

		/* dphimin cut */

		if ((electrons_n + muons_n) == 0) {
			float min = std::numeric_limits<float>::max();
			for (size_t i = 0; i < 4 && i < jets_pt->size(); i++) {
				float jphi = jets_phi->at(i);
				float dphi = abs(
					TVector2::Phi_mpi_pi(jphi - mettst_phi)
					);
				if (dphi < min)
					min = dphi;
			}
			if (min < 0.4)
				continue;
		}
		h_cutflow->Fill(5);

		if (ht_filter && gen_filt_ht > 600)
			continue;
		h_cutflow->Fill(6);

		if (met_filter && gen_filt_met > 200)
			continue;
		h_cutflow->Fill(7);

		tree->Fill();
	}
	return h_cutflow;
}
		
		

int main(int argc, const char **argv)
{
	/* Parse the argv */

	if (argc < 4)
		ERROR("usage: preselect <dsid> <output> <input>...");
	ULong64_t dsid = std::strtoul(argv[1], nullptr, 10);
	INFO("DSID: " << dsid);
	const char *output_path = argv[2];
	INFO("Output path: " << output_path);
	std::vector<const char *> input_paths;
	for (int i = 3; i < argc; i++) {
		input_paths.push_back(argv[i]);
		INFO("Input path #" << (i - 3) << ": " << argv[i]);
	}

	/* Get the input chain */
	
	TChain chain("nominal");
	for (const char *path : input_paths) {
		// 0 to force reading the header
		// this ensures a meaningful return value
		if (!chain.Add(path, 0))
			ERROR("Unable to add file to TChain");
	}

	/* Compute the scale-factor */
	
	double weight = 0;
	double xsec = 0;
	for (const char *path : input_paths) {
		TFile *file = TFile::Open(path);
		TH1 *cutflow = (TH1*)file->Get("cut_flow");
		weight += cutflow->GetBinContent(2);
		TH1 *hxsec = (TH1*)file->Get("cross_section");
		xsec = hxsec->GetBinContent(1) / hxsec->GetEntries();
		delete cutflow;
		delete hxsec;
		file->Close();
		delete file;
	}
        float weight_1ifb = 1000.0 * xsec / weight;
	INFO("Scale factor: " << weight_1ifb);

	/* Open the output file */

	TFile outfile(output_path, "CREATE");
	if (outfile.IsZombie())
		ERROR("Unable to create output TFile");

	/* Thin the input chain */

	INFO("Thinning the input tree...");
	do_thinning(chain);

	/* Clone the input chain structure */

	INFO("Cloning the input chain structure");
	TTree *output_tree = chain.CloneTree(0);
	   
	/* Augment the output tree */

	INFO("Creating additional branches...");
	output_tree->Branch("dsid", &dsid, "dsid/l");
	output_tree->Branch("weight_1ifb", &weight_1ifb, "dsid/F");
	
	/* Skim the output tree */

	INFO("Skimming the output tree...");
	bool ht_filter = (dsid == 410000);
	bool met_filter = (dsid == 410013) || (dsid == 410014);
	INFO("Applying truth HT filter: " << (ht_filter? "YES" : "NO"));
	INFO("Applying truth MET filter: " << (met_filter? "YES" : "NO"));
	TH1D *cutflow = do_skimming(chain, output_tree, ht_filter, met_filter);

	Long64_t pas = output_tree->GetEntries();
	Long64_t tot = chain.GetEntries();
	INFO("Retained " << pas << " events out of " << tot);
	INFO("Efficiency: " << static_cast<Double_t>(pas) / tot);
	info_cutflow(cutflow, ht_filter, met_filter);
}
