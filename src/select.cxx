#include <algorithm>
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
	vector<int> *muons_isBad;
	vector<int> *muons_isCosmic;
 	vector<int> *muons_isSignal;
	vector<float> *electrons_pt;
	vector<float> *electrons_eta;
	vector<float> *electrons_phi;
	vector<float> *electrons_e;
 	vector<int> *electrons_isSignal;
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
	int run_number;
	int event_number;
	float gen_filt_ht;
	float gen_filt_met;
	vector<bool> *trigger;
};

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
	CONNECT(muons_isBad);
	CONNECT(muons_isCosmic);
	CONNECT(muons_isSignal);
	CONNECT(electrons_pt);
	CONNECT(electrons_eta);
	CONNECT(electrons_phi);
	CONNECT(electrons_e);
	CONNECT(electrons_isSignal);
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
	CONNECT(run_number);
	CONNECT(event_number);
	CONNECT(gen_filt_ht);
	CONNECT(gen_filt_met);
	CONNECT(trigger);
#undef CONNECT
}

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
	/* weight */
	double weight;
	/* metadata */
	double event_number;
	double run_number;
	double meff;
	double mt;
	double mtb;
	double mjsum;
	double nb;
	double nlepton;

	OutData(int n_small=12, int n_large=4, int n_lepton=4);
};

OutData::OutData(int n_small, int n_large, int n_lepton)
{
	small_R_jets_pt.resize(n_small);
	small_R_jets_eta.resize(n_small);
	small_R_jets_phi.resize(n_small);
	small_R_jets_m.resize(n_small);

	large_R_jets_pt.resize(n_large);
	large_R_jets_eta.resize(n_large);
	large_R_jets_phi.resize(n_large);
	large_R_jets_m.resize(n_large);

	leptons_pt.resize(n_lepton);
	leptons_eta.resize(n_lepton);
	leptons_phi.resize(n_lepton);
	leptons_m.resize(n_lepton);
}

template <typename T>
std::string appendT(const string& prefix, const T& thing)
{
	std::ostringstream stream;
	stream << prefix << '_' << thing;
	return stream.str();
}

void connect_outdata(OutData &outdata, TTree &tree)
{
#define CONNECT_I(b,i) key = appendT(#b, i); \
	tree.Branch(#b, outdata.b.data() + i)
#define CONNECT(b) outdata.b = 0; tree.Branch(#b, &(outdata.b))

	std::string key;

	for (size_t i = 0; i < outdata.small_R_jets_pt.size(); i++) {
		CONNECT_I(small_R_jets_pt, i);
		CONNECT_I(small_R_jets_eta, i);
		CONNECT_I(small_R_jets_phi, i);
		CONNECT_I(small_R_jets_m, i);
	}
	for (size_t i = 0; i < outdata.large_R_jets_pt.size(); i++) {
		CONNECT_I(large_R_jets_pt, i);
		CONNECT_I(large_R_jets_eta, i);
		CONNECT_I(large_R_jets_phi, i);
		CONNECT_I(large_R_jets_m, i);
	}
	for (size_t i = 0; i < outdata.leptons_pt.size(); i++) {
		CONNECT_I(leptons_pt, i);
		CONNECT_I(leptons_eta, i);
		CONNECT_I(leptons_phi, i);
		CONNECT_I(leptons_m, i);
	}

	CONNECT(met_mag);
	CONNECT(met_phi);
	CONNECT(weight);
	CONNECT(event_number);
	CONNECT(run_number);
	CONNECT(meff);
	CONNECT(mt);
	CONNECT(mtb);
	CONNECT(mjsum);
	CONNECT(nb);
	CONNECT(nlepton);
#undef CONNECT_I
#undef CONNECT
}

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


TLorentzVector make_tlv(double pt, double eta, double phi, double e)
{
	TLorentzVector tlv;
	tlv.SetPtEtaPhiE(pt,eta,phi,e);
	return tlv;
}

bool compare_tlv(TLorentzVector v1, TLorentzVector v2)
{
	return v1.Pt() > v2.Pt();
}

vector<TLorentzVector> get_leptons(InData& data)
{
	vector<TLorentzVector> leptons;

	for (size_t i = 0; i < data.electrons_pt->size(); ++i) {
		bool isSignal = data.electrons_isSignal->at(i);
		if (isSignal) {
			double pt = data.electrons_pt->at(i);
			double eta = data.electrons_eta->at(i);
			double phi = data.electrons_phi->at(i);
			double e = data.electrons_e->at(i);
			leptons.push_back(make_tlv(pt,eta,phi,e));
		}
	}

	for (size_t i = 0; i < data.muons_pt->size(); ++i) {
		bool isCosmic = data.muons_isCosmic->at(i);
		bool isBad = data.muons_isBad->at(i);
		bool isSignal = data.muons_isSignal->at(i);
		if (!isCosmic && !isBad && isSignal) {
			double pt = data.muons_pt->at(i);
			double eta = data.muons_eta->at(i);
			double phi = data.muons_phi->at(i);
			double e = data.muons_e->at(i);
			leptons.push_back(make_tlv(pt,eta,phi,e));
		}
	}

	sort(leptons.begin(),leptons.end(),compare_tlv);

	return leptons;
}

bool compare_tlv_in_pair(pair<TLorentzVector,bool> a,
			 pair<TLorentzVector,bool> b)
{
	return compare_tlv(a.first, b.first);
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

Event get_event(InData& data)
{
	double weight =
		data.weight_mc *
		data.weight_btag *
		data.weight_elec *
		data.weight_muon;

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
		  data.trigger->at(1)); // HLT_xe80_tc_lcw_L1XE50
	return evt;
}

#define PT_AT(i,vect) ((vect.size() > i)? vect.at(i).Pt() : 0)

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

void fill_output_vectors(std::vector<TLorentzVector>& inputs,
		    std::vector<double>& pt,
		    std::vector<double>& eta,
		    std::vector<double>& phi,
		    std::vector<double>& m)
{
	for(size_t i = 0; i < pt.size(); i++) {
		bool zero = i >= inputs.size();
		pt.at(i) = zero ? 0 : inputs.at(i).Pt();
		eta.at(i) = zero ? 0 : inputs.at(i).Pt();
		phi.at(i) = zero ? 0 : inputs.at(i).Pt();
		m.at(i) = zero ? 0 : inputs.at(i).M();
	}
}

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
	outdata.nb = evt.bjets.size();
	outdata.nlepton = evt.leptons.size();
}


double get_scale_factor(int nfile, char *paths[], double xsec)
{
	double weight = 0;
	for (int i = 0; i < nfile; ++i) {
		TFile file(paths[i]);
		TH1 *cutflow = (TH1*)file.Get("cut_flow");
		weight += cutflow->GetBinContent(2);
		if (xsec == 0) {
			TH1 *hxsec = (TH1*)file.Get("cross_section");
			xsec = hxsec->GetBinContent(1) / hxsec->GetEntries();
		}
	}

	return 1000.0 * xsec / weight;
}

bool good_event(Event &event, bool met_filter_under200, bool ht_filter_under600)
{
	return
		   (!met_filter_under200 || event.met_filter < 200)
		&& (!ht_filter_under600  || event.met_filter < 600)
		&& (event.trigger)
		&& (event.jets.size() >= 4)
		&& (event.bjets.size() >= 2);
}

int main(int argc, char *argv[])
{
	bool met_filter_under200 = false;
	bool ht_filter_under600 = false;
	char *outpath;
	double xsec;
	int in_start;

	if (argc < 4)
		return -1;

	if (argv[1][0] == '-') {
		outpath = argv[2];
		xsec = atof(argv[3]);
		in_start = 4;
		met_filter_under200 = argv[1][1] == 'f';
		ht_filter_under600 = argv[1][1] == 'F';
	} else {
		outpath = argv[1];
		xsec = atof(argv[2]);
		in_start = 3;
	}

	TChain chain("nominal");
	for (int i = in_start; i < argc; ++i)
		chain.Add(argv[i]);

	string outpathstr(outpath);
	// VERSION
	outpathstr += (".7.root");
	TFile outfile(outpathstr.c_str(), "CREATE");

	InData indata;
	connect_indata(indata,chain);

	TTree outtree("tree","tree");
	OutData outdata;
	connect_outdata(outdata, outtree);


	double scale = get_scale_factor(argc - in_start, argv + in_start, xsec);

	for (Long64_t i = 0; i < chain.GetEntries(); ++i) {
		chain.GetEntry(i);
		Event evt = get_event(indata);
		if (good_event(evt, met_filter_under200, ht_filter_under600)) {
			fill_outdata(evt,outdata,scale);
			outtree.Fill();
		}
	}

	outfile.Write(0,TObject::kWriteDelete);
}
