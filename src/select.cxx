/* select.cxx: Code to select and reformat events from small ntuples
 * produced by the MBJ framework.
 *
 * First, events are pulled from the MBJ ntuples in a "raw"
 * vector-based representation (InData) that reprensents directly the
 * layout of the needed variables in the ntuple. This raw
 * reprensentation is then translated into an intermediate
 * T(Lorentz)Vector based reprensentation (Event) more suitable to
 * physics calculations via the get_event() function. Using this
 * reprensentation, events are selected via the good_event() function
 * and finally transformed to the final, vector-based format (OutData)
 * which maps directly to a flat output format.
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
using namespace std;

#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TLorentzVector.h>
#include <TTree.h>


/*******************************************************************************
 * Data structure declarations
 */


/* This structure holds data pulled from ROOT files produced by the
 * MBJ framework
 */
struct InData {
	vector<float> *jets_pt;
	vector<float> *jets_eta;
	vector<float> *jets_phi;
	vector<float> *jets_e;
	vector<int> *jets_isb_77;
	vector<float> *muons_pt;
	vector<float> *muons_eta;
	vector<float> *muons_phi;
	vector<float> *muons_e;
	vector<float> *electrons_pt;
	vector<float> *electrons_eta;
	vector<float> *electrons_phi;
	vector<float> *electrons_e;
  	vector<float> *rc_R08PT10_jets_pt;
	vector<float> *rc_R08PT10_jets_eta;
	vector<float> *rc_R08PT10_jets_phi;
	vector<float> *rc_R08PT10_jets_e;
	vector<float> *rc_R08PT10_jets_m;
	float mettst;
	float mettst_phi;
	double weight_mc;
	double weight_btag;
	double weight_elec;
	double weight_muon;
    	double weight_pu;
	double weight_jvt;
	double weight_WZ_2_2;
	double weight_ttbar_NNLO;
	double weight_ttbar_NNLO_1L;
	int run_number;
	int event_number;
	float gen_filt_ht;
	float gen_filt_met;
        int pass_MET;
};


/* Connect a ttree to an InData structure
 *
 * Arguments:
 *     -- data: the InData structure to be connected
 *     -- chain: the TTree to connect to
 * Returns: Nothing
 */
void connect_indata(InData &data, TTree &chain)
{
	chain.SetBranchStatus("*", 0);
#define CONNECT(b) data.b = 0; chain.SetBranchStatus(#b,1); chain.SetBranchAddress(#b,&(data.b))
	CONNECT(jets_pt);
	CONNECT(jets_eta);
	CONNECT(jets_phi);
	CONNECT(jets_e);
	CONNECT(jets_isb_77);
	CONNECT(muons_pt);
	CONNECT(muons_eta);
	CONNECT(muons_phi);
	CONNECT(muons_e);
	CONNECT(electrons_pt);
	CONNECT(electrons_eta);
	CONNECT(electrons_phi);
	CONNECT(electrons_e);
	CONNECT(rc_R08PT10_jets_pt);
	CONNECT(rc_R08PT10_jets_eta);
	CONNECT(rc_R08PT10_jets_phi);
	CONNECT(rc_R08PT10_jets_e);
	CONNECT(rc_R08PT10_jets_m);
	CONNECT(mettst);
	CONNECT(mettst_phi);
	CONNECT(weight_mc);
	CONNECT(weight_btag);
	CONNECT(weight_elec);
	CONNECT(weight_muon);
	CONNECT(weight_pu);
	CONNECT(weight_jvt);
	CONNECT(weight_WZ_2_2);
	CONNECT(weight_ttbar_NNLO);
	CONNECT(weight_ttbar_NNLO_1L);
	CONNECT(run_number);
	CONNECT(event_number);
	CONNECT(gen_filt_ht);
	CONNECT(gen_filt_met);
	CONNECT(pass_MET);
#undef CONNECT
}


/* Structure to hold data in the needed output format. Note that even
 * though vectors are used here to easily make trees with a different
 * about of variables, the output tree will be completely flat.
 */
struct OutData {
	/* inputs for neural network */
	std::vector<double> small_R_jets_pt;
	std::vector<double> small_R_jets_eta;
	std::vector<double> small_R_jets_phi;
	std::vector<double> small_R_jets_m;
	std::vector<double> small_R_jets_isb;
	std::vector<double> large_R_jets_pt;
	std::vector<double> large_R_jets_eta;
	std::vector<double> large_R_jets_phi;
	std::vector<double> large_R_jets_m;
	std::vector<double> leptons_pt;
	std::vector<double> leptons_eta;
	std::vector<double> leptons_phi;
	std::vector<double> leptons_m;
	double met_mag;
	double met_phi;
        //double m_gluino;
	//double m_lsp;

	/* metadata */
	double weight;
	double event_number;
	double run_number;
	double meff;
	double mt;
	double mtb;
	double mjsum;
	double nb77;
	double nlepton;
	double njet30;
	double dphimin4j;
	double met;

	OutData(int n_small, int n_large, int n_lepton);
};


/* Constructor that initialize all vectors to requested size
 *
 * Arguments:
 *   -- n_small: number of small-R jets
 *   -- n_large: number of large-R jets
 *   -- n_lepton: number of leptons
 * Returns:
 *   an empty OutData structure
 */
OutData::OutData(int n_small, int n_large, int n_lepton)
{
	small_R_jets_pt.resize(n_small);
	small_R_jets_eta.resize(n_small);
	small_R_jets_phi.resize(n_small);
	small_R_jets_m.resize(n_small);
	small_R_jets_isb.resize(n_small);

	large_R_jets_pt.resize(n_large);
	large_R_jets_eta.resize(n_large);
	large_R_jets_phi.resize(n_large);
	large_R_jets_m.resize(n_large);

	leptons_pt.resize(n_lepton);
	leptons_eta.resize(n_lepton);
	leptons_phi.resize(n_lepton);
	leptons_m.resize(n_lepton);
}


/* Helper function to append a string representation of something to a
 * prefix string
 *
 * Arguments:
 *   -- prefix: the prefix string
 *   -- thing: the thing to append
 * Returns:
 *   a string in the "<prefix>_<string repr of `thing`>" format
 */
template <typename T>
std::string appendT(const string& prefix, const T& thing)
{
	std::ostringstream stream;
	stream << prefix << '_' << thing;
	return stream.str();
}


/* Connect OutData struct to output TTree
 *
 * Arguments:
 *   -- outdata: The OutData struct to connect
 *   -- tree: The TTree to connect to
 */
void connect_outdata(OutData &outdata, TTree &tree)
{
#define CONNECT_I(p,b,i) do {std::string key = appendT(p #b, i); \
		tree.Branch(key.c_str(), outdata.b.data() + i); } while (0)
#define CONNECT(p, b) outdata.b = 0; tree.Branch(p #b, &(outdata.b))

	for (size_t i = 0; i < outdata.small_R_jets_pt.size(); i++) {
		CONNECT_I("I_", small_R_jets_pt, i);
		CONNECT_I("I_", small_R_jets_eta, i);
		CONNECT_I("I_", small_R_jets_phi, i);
		CONNECT_I("I_", small_R_jets_m, i);
		CONNECT_I("I_", small_R_jets_isb, i);
	}
	for (size_t i = 0; i < outdata.large_R_jets_pt.size(); i++) {
		CONNECT_I("I_", large_R_jets_pt, i);
		CONNECT_I("I_", large_R_jets_eta, i);
		CONNECT_I("I_", large_R_jets_phi, i);
		CONNECT_I("I_", large_R_jets_m, i);
	}
	for (size_t i = 0; i < outdata.leptons_pt.size(); i++) {
		CONNECT_I("I_", leptons_pt, i);
		CONNECT_I("I_", leptons_eta, i);
		CONNECT_I("I_", leptons_phi, i);
		CONNECT_I("I_", leptons_m, i);
	}

	CONNECT("I_", met_mag);
	CONNECT("I_", met_phi);
	// CONNECT("I_", m_gluino);
	// CONNECT("I_", m_lsp);
	CONNECT("M_", weight);
	CONNECT("M_", event_number);
	CONNECT("M_", run_number);
	CONNECT("M_", meff);
	CONNECT("M_", mt);
	CONNECT("M_", mtb);
	CONNECT("M_", mjsum);
	CONNECT("M_", nb77);
	CONNECT("M_", nlepton);
	CONNECT("M_", njet30);
	CONNECT("M_", dphimin4j);
	CONNECT("M_", met);

#undef CONNECT_I
#undef CONNECT
}


/* Intermediate representation of an event to make computation on
 * physics objects easier.
 */
struct Event {

	Event(vector<TLorentzVector> &&leptons_,
	      vector<pair<TLorentzVector,bool>> &&jets_,
	      vector<TLorentzVector> &&bjets_,
	      vector<TLorentzVector> &&largejets_,
	      TVector2 &&met_,
	      int run_number_,
	      int event_number_,
	      double weight_,
	      double met_filter_,
	      double ht_filter_,
	      bool trigger_)
		:
		leptons(leptons_),
		jets(jets_),
		bjets(bjets_),
		largejets(largejets_),
		met(met_),
		run_number(run_number_),
		event_number(event_number_),
		weight(weight_),
		met_filter(met_filter_),
		ht_filter(ht_filter_),
		trigger(trigger_)
		{};

	vector<TLorentzVector> leptons;
	vector<pair<TLorentzVector,bool>> jets;
	vector<TLorentzVector> bjets;
	vector<TLorentzVector> largejets;

	TVector2 met;
	int run_number;
	int event_number;
	double weight;
	double met_filter;
	double ht_filter;
	bool trigger;
};


/*******************************************************************************
 * TLorentzVector helper functions
 */


/* Make a TLorentzVector in one call from pt, eta, phi and energy */
TLorentzVector make_tlv(double pt, double eta, double
phi, double e) { TLorentzVector tlv; tlv.SetPtEtaPhiE(pt,eta,phi,e);
return tlv; }


/* Used to sort containers of TLorentzVector by decreasing pT */
bool compare_tlv(TLorentzVector v1, TLorentzVector v2)
{
	return v1.Pt() > v2.Pt();
}


/* Used to sort containers of TLorentzVector by decreasing pT, when
 * tagging information is included
 */
bool compare_tlv_in_pair(pair<TLorentzVector,bool> a,
			 pair<TLorentzVector,bool> b)
{
	return compare_tlv(a.first, b.first);
}


/*******************************************************************************
 * Helpers for the InData -> Event conversion
 */

vector<TLorentzVector> get_leptons(InData& data)
{
	vector<TLorentzVector> leptons;

	for (size_t i = 0; i < data.electrons_pt->size(); ++i) {
	        double pt = data.electrons_pt->at(i);
		double eta = data.electrons_eta->at(i);
		double phi = data.electrons_phi->at(i);
		double e = data.electrons_e->at(i);
		leptons.push_back(make_tlv(pt,eta,phi,e));
	}

	for (size_t i = 0; i < data.muons_pt->size(); ++i) {
	        double pt = data.muons_pt->at(i);
		double eta = data.muons_eta->at(i);
		double phi = data.muons_phi->at(i);
		double e = data.muons_e->at(i);
		leptons.push_back(make_tlv(pt,eta,phi,e));
	}

	sort(leptons.begin(),leptons.end(),compare_tlv);

	return leptons;
}

vector<pair<TLorentzVector,bool>> get_jets(InData &data)
{
	vector<pair<TLorentzVector,bool> > jets;

	for (size_t i = 0; i < data.jets_pt->size(); i++) {
		double pt = data.jets_pt->at(i);
		double eta = data.jets_eta->at(i);
		double phi = data.jets_phi->at(i);
		double e = data.jets_e->at(i);
		double isb = data.jets_isb_77->at(i) && pt > 30 && abs(eta) < 2.5;
		if (pt > 30 && abs(eta) < 2.8) {
			pair<TLorentzVector,bool> pair(
				make_tlv(pt,eta,phi,e),
				isb);
			jets.push_back(move(pair));
		}
	}

	sort(jets.begin(), jets.end(), compare_tlv_in_pair);

	return jets;
}

vector<TLorentzVector> get_bjets(InData &data)
{
	vector<TLorentzVector> jets;

	for (size_t i = 0; i < data.jets_pt->size(); i++) {
		double pt = data.jets_pt->at(i);
		double eta = data.jets_eta->at(i);
		double phi = data.jets_phi->at(i);
		double e = data.jets_e->at(i);
		bool isb = data.jets_isb_77->at(i) == 1;

		if (pt > 30 && abs(eta) < 2.5 && isb)
			jets.push_back(make_tlv(pt,eta,phi,e));
	}

	sort(jets.begin(), jets.end(), compare_tlv);

	return jets;
}

vector<TLorentzVector> get_largeR_jets(InData &data)
{
	vector<TLorentzVector> jets;

	for (size_t i = 0; i < data.rc_R08PT10_jets_pt->size(); i++) {
		double pt = data.rc_R08PT10_jets_pt->at(i);
		double eta = data.rc_R08PT10_jets_eta->at(i);
		double phi = data.rc_R08PT10_jets_phi->at(i);
		double e = data.rc_R08PT10_jets_e->at(i);
		if (pt > 100 && abs(eta) < 2.0)
			jets.push_back(make_tlv(pt,eta,phi,e));
	}

	sort(jets.begin(), jets.end(), compare_tlv);

	return jets;
}

TVector2 get_met(InData &data)
{
	TVector2 v;
	v.SetMagPhi(data.mettst, data.mettst_phi);
	return v;
}


/* put everything together */
Event get_event(InData& data)
{
	double weight =
		data.weight_mc *
		data.weight_btag *
		data.weight_elec *
		data.weight_muon *
	    	data.weight_pu *
		data.weight_jvt *
		data.weight_WZ_2_2 *
		data.weight_ttbar_NNLO *
		data.weight_ttbar_NNLO_1L;


	Event evt(get_leptons(data),
		  get_jets(data),
		  get_bjets(data), // OP = 77%
		  get_largeR_jets(data),
		  get_met(data),
		  data.run_number,
		  data.event_number,
		  weight,
		  data.gen_filt_met,
		  data.gen_filt_ht,
		  data.pass_MET);
	return evt;
}


/*******************************************************************************
 * Functions to compute observables
 */

double calc_meff(vector<TLorentzVector>& jets,
		 vector<TLorentzVector>& leptons,
		 TVector2& met)
{
	double meff = 0;
	for (TLorentzVector v : jets)
		meff += v.Pt();
	for (TLorentzVector v : leptons)
		meff += v.Pt();
	meff += met.Mod();
	return meff;
}

double calc_mt(vector<TLorentzVector>& leptons, TVector2& met)
{
	if (leptons.size() == 0)
		return 0;

	TLorentzVector lep0 = leptons.at(0);
	return sqrt(2*lep0.Pt()*met.Mod()*(1 - cos(lep0.Phi() - met.Phi())));
}

double calc_mt_min_bjets(vector<TLorentzVector> &bjets, TVector2 &met)
{
	double mt_min = numeric_limits<double>::max();

	for (size_t i = 0; i < bjets.size() && i < 3; i++) {
		TLorentzVector b = bjets.at(i);
		double mt =
			pow(met.Mod() + b.Pt(), 2) -
			pow(met.Px()  + b.Px(), 2) -
			pow(met.Py()  + b.Py(), 2);
		mt = (mt >= 0)? sqrt(mt) : sqrt(-mt);
		if (mt < mt_min)
			mt_min = mt;
	}

	return (bjets.size() > 0)? mt_min : 0;
}

double calc_mjsum(vector<TLorentzVector>& largejets)
{
	double sum = 0;
	for (size_t i = 0; i < 4 && i < largejets.size(); i++)
		sum += largejets.at(i).M();
	return sum;
}

double calc_dphi_min_4j(vector<pair<TLorentzVector,bool>>& jets, TVector2 met)
{
	double min = std::numeric_limits<double>::max();
	for (size_t i = 0; i < 4 && i < jets.size(); i++) {
		TLorentzVector j = jets.at(i).first;
		double dphi = abs(TVector2::Phi_mpi_pi(j.Phi() - met.Phi()));
		if (dphi < min)
			min = dphi;
	}
	return min;
}

double calc_njet(vector<pair<TLorentzVector,bool>>& jets, double ptcut)
{
	double njet = 0;

	for (size_t i = 0; i < jets.size(); i++) {
		if (jets.at(i).first.Pt() > ptcut) {
			njet += 1;
		}
	}

	return njet;
}


/*******************************************************************************
 * Helpers for the Event -> OutData conversion
 */


/* Unwrap a vector of TLorentzVector into separate flat vectors */
void fill_output_vectors(std::vector<TLorentzVector>& inputs,
		    std::vector<double>& pt,
		    std::vector<double>& eta,
		    std::vector<double>& phi,
		    std::vector<double>& m)
{
	for(size_t i = 0; i < pt.size(); i++) {
		bool zero = i >= inputs.size();
		pt.at(i) = zero ? 0 : inputs.at(i).Pt();
		eta.at(i) = zero ? 0 : inputs.at(i).Eta();
		phi.at(i) = zero ? 0 : inputs.at(i).Phi();
		m.at(i) = zero ? 0 : inputs.at(i).M();
	}
}

/* Unwrap a vector of TLorentzVector + tagging information into
 * separate flat vectors
 */
void fill_output_vectors(std::vector<pair<TLorentzVector,bool>>& inputs,
		    std::vector<double>& pt,
		    std::vector<double>& eta,
		    std::vector<double>& phi,
		    std::vector<double>& m,
		    std::vector<double> &tag)
{
	for(size_t i = 0; i < pt.size(); i++) {
		bool zero = i >= inputs.size();
		pt.at(i) = zero ? 0 : inputs.at(i).first.Pt();
		eta.at(i) = zero ? 0 : inputs.at(i).first.Eta();
		phi.at(i) = zero ? 0 : inputs.at(i).first.Phi();
		m.at(i) = zero ? 0 : inputs.at(i).first.M();
		tag.at(i) = zero ? 0 : inputs.at(i).second;
	}
}


/* Event -> OutData conversion
 *
 * Arguments:
 *   -- event: The Event structure to convert
 *   -- outdata: The OutData struct to convert into
 *   -- scale: quantity by which to scale the event weight.
 *             can be used to scale the sum of the weights to the
 *             desired xsec.
 * Returns:
 *   Nothing
 */
void fill_outdata(Event &evt, OutData &outdata, double scale)
{
	fill_output_vectors(evt.jets,
			    outdata.small_R_jets_pt,
			    outdata.small_R_jets_eta,
			    outdata.small_R_jets_phi,
			    outdata.small_R_jets_m,
			    outdata.small_R_jets_isb);

	fill_output_vectors(evt.largejets,
			    outdata.large_R_jets_pt,
			    outdata.large_R_jets_eta,
			    outdata.large_R_jets_phi,
			    outdata.large_R_jets_m);

	fill_output_vectors(evt.leptons,
			    outdata.leptons_pt,
			    outdata.leptons_eta,
			    outdata.leptons_phi,
			    outdata.leptons_m);

	vector<TLorentzVector> jets_tlv_only;
	for (auto p : evt.jets)
		jets_tlv_only.push_back(p.first);

	outdata.met_mag = evt.met.Mod();
	outdata.met_phi = evt.met.Phi();
	outdata.weight = evt.weight * scale;
	outdata.event_number = evt.event_number;
	outdata.run_number = evt.run_number;
	outdata.meff = calc_meff(jets_tlv_only, evt.leptons, evt.met);
	outdata.mt = calc_mt(evt.leptons, evt.met);
	outdata.mtb = calc_mt_min_bjets(evt.bjets, evt.met);
	outdata.mjsum = calc_mjsum(evt.largejets);
	outdata.nb77 = evt.bjets.size();
	outdata.nlepton = evt.leptons.size();
	outdata.njet30 = calc_njet(evt.jets, 30);
	outdata.dphimin4j = calc_dphi_min_4j(evt.jets, evt.met);
	outdata.met = outdata.met_mag;
}


/* Get the 1 fb scale factor from the xsec & cutflow histograms stored
 * in the MBJ framework-produced TFiles */
double get_scale_factor(int nfile, char *paths[])
{
	double weight = 0;
	double xsec = 0;
	for (int i = 0; i < nfile; ++i) {
		TFile *file = TFile::Open(paths[i]);
		TH1 *cutflow = (TH1*)file->Get("cut_flow");
		weight += cutflow->GetBinContent(2);
		TH1 *hxsec = (TH1*)file->Get("cross_section");
		xsec = hxsec->GetBinContent(1) / hxsec->GetEntries();
		file->Close();
	}

	return 1000.0 * xsec / weight;
}


/* Decide if good event or not, with met and ht cuts */
bool good_event(Event &event, double met_max, double ht_max)
{
	bool good = (event.met_filter < met_max)
		&& (event.ht_filter < ht_max)
		&& (event.trigger)
		&& (event.jets.size() >= 4)
		&& (event.bjets.size() >= 2)
		&& (event.met.Mod() > 200);

	if (event.leptons.size() == 0)
		good = good && (calc_dphi_min_4j(event.jets, event.met) > 0.4);

	return good;
}

// usage: select output nsmall nlarge nlepton met_max ht_max target m_g m_l inputs...
//        ^0     ^1     ^2     ^3     ^4     ^5      ^6      ^7     ^8  ^9  ^10...
int main(int argc, char *argv[])
{

	if (argc < 11) {
		fprintf(stderr, "ERROR: too few arguments\n");
		fprintf(stderr, "usage: select output nsmall nlarge nlepton met_max ht_max target m_g m_l inputs...\n");
		return 1;
	}

	TChain chain("nominal");
	for (int i = 10; i < argc; ++i) {
		 // 0 to force reading the header
		if (!chain.Add(argv[i], 0)) {
			fprintf(stderr, "ERROR: %s: unable to add\n", argv[i]);
			return 1;
		}
	}

	TFile outfile(argv[1], "CREATE");
	if (outfile.IsZombie()) {
		fprintf(stderr, "ERROR: unable to open output file %s\n", argv[1]);
		return 1;
	}

	int nsmall = atoi(argv[2]);
	int nlarge = atoi(argv[3]);
	int nlepton = atoi(argv[4]);
	double met_max = atof(argv[5]);
	double ht_max = atof(argv[6]);
	double target = atof(argv[7]);
	// double m_g = atof(argv[8]);
	// double m_l = atof(argv[9]);

	InData indata;
	connect_indata(indata,chain);

	TTree outtree("NNinput","");
	OutData outdata(nsmall, nlarge, nlepton);
	connect_outdata(outdata, outtree);
	// outdata.m_gluino = m_g;
	// outdata.m_lsp = m_l;
	// TODO move this
	outtree.Branch("L_target", &target);

	double scale = get_scale_factor(argc - 10, argv + 10);

	for (Long64_t i = 0; i < chain.GetEntries(); ++i) {
		chain.GetEntry(i);
		Event evt = get_event(indata);
		if (good_event(evt, met_max, ht_max)) {
			fill_outdata(evt,outdata,scale);
			outtree.Fill();
		}
	}

	outfile.Write(0,TObject::kWriteDelete);
}
