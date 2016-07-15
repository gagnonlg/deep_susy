#include <algorithm>
#include <limits>
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
	vector<int> *jets_isBad;
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
	double weight_pu;
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
	CONNECT(jets_isBad);
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
	CONNECT(weight_pu);
	CONNECT(run_number);
	CONNECT(event_number);
	CONNECT(gen_filt_ht);
	CONNECT(gen_filt_met);
	CONNECT(trigger);
#undef CONNECT
}

struct OutData {
	double meff;
	double met;
	double mt;
	double jet_1_pt;
	double jet_4_pt;
	double jet_5_pt;
	double jet_6_pt;
	double jet_7_pt;
	double jet_8_pt;
	double n_b_60;
	double n_b_70;
	double n_b_85;
	double mt_min_b_60;
	double mt_min_b_70;
	double mt_min_b_85;
	double veryloose_top_1_pt;
	double veryloose_top_2_pt;
	double loose_top_1_pt;
	double loose_top_2_pt;
	double tight_top_1_pt;
	double tight_top_2_pt;
	double smoothloose_top_1_pt;
	double smoothloose_top_2_pt;
	double smoothtight_top_1_pt;
	double smoothtight_top_2_pt;
	double loose_top_b60_1_pt;
	double loose_top_b60_2_pt;
	double tight_top_b60_1_pt;
	double tight_top_b60_2_pt;
	double loose_top_b70_1_pt;
	double loose_top_b70_2_pt;
	double tight_top_b70_1_pt;
	double tight_top_b70_2_pt;
	double loose_top_b85_1_pt;
	double loose_top_b85_2_pt;
	double tight_top_b85_1_pt;
	double tight_top_b85_2_pt;
	double event_number;
	double run_number;
	double weight;
	double gen_filt_met;
	double gen_filt_ht;
	double n_lepton;

	/* needed for histograms */
	vector<double> *jets_pt;
};

void connect_outdata(OutData &outdata, TTree &tree)
{
#define BRANCH(b) outdata.b = 0; tree.Branch(#b,&(outdata.b))
	BRANCH(meff);
	BRANCH(met);
	BRANCH(mt);
	BRANCH(jet_1_pt);
	BRANCH(jet_4_pt);
	BRANCH(jet_5_pt);
	BRANCH(jet_6_pt);
	BRANCH(jet_7_pt);
	BRANCH(jet_8_pt);
	BRANCH(n_b_60);
	BRANCH(n_b_70);
	BRANCH(n_b_85);
	BRANCH(mt_min_b_60);
	BRANCH(mt_min_b_70);
	BRANCH(mt_min_b_85);
	BRANCH(veryloose_top_1_pt);
	BRANCH(veryloose_top_2_pt);
	BRANCH(loose_top_1_pt);
	BRANCH(loose_top_2_pt);
	BRANCH(tight_top_1_pt);
	BRANCH(tight_top_2_pt);
	BRANCH(smoothloose_top_1_pt);
	BRANCH(smoothloose_top_2_pt);
	BRANCH(smoothtight_top_1_pt);
	BRANCH(smoothtight_top_2_pt);
	BRANCH(loose_top_b60_1_pt);
	BRANCH(loose_top_b60_2_pt);
	BRANCH(tight_top_b60_1_pt);
	BRANCH(tight_top_b60_2_pt);
	BRANCH(loose_top_b70_1_pt);
	BRANCH(loose_top_b70_2_pt);
	BRANCH(tight_top_b70_1_pt);
	BRANCH(tight_top_b70_2_pt);
	BRANCH(loose_top_b85_1_pt);
	BRANCH(loose_top_b85_2_pt);
	BRANCH(tight_top_b85_1_pt);
	BRANCH(tight_top_b85_2_pt);
	BRANCH(event_number);
	BRANCH(run_number);
	BRANCH(weight);
	BRANCH(jets_pt);
	BRANCH(gen_filt_ht);
	BRANCH(gen_filt_met);
	BRANCH(n_lepton);
#undef BRANCH
}

vector<TLorentzVector> get_top_b_match(vector<TLorentzVector> &tops, vector<TLorentzVector> & bjets)
{
	vector<TLorentzVector> matched;

	for (TLorentzVector t : tops) {
		for (TLorentzVector b : bjets) {
			if (t.DeltaR(b) <= 1.0) {
				matched.push_back(t);
				break;
			}
		}
	}
	return matched;
}


struct Event {

	Event(vector<TLorentzVector> &&leptons_,
	      vector<TLorentzVector> &&jets_,
	      vector<TLorentzVector> &&bjets_60_,
	      vector<TLorentzVector> &&bjets_70_,
	      vector<TLorentzVector> &&bjets_85_,
	      vector<TLorentzVector> &&veryloose_tops_,
	      vector<TLorentzVector> &&loose_tops_,
	      vector<TLorentzVector> &&tight_tops_,
	      vector<TLorentzVector> &&smoothloose_tops_,
	      vector<TLorentzVector> &&smoothtight_tops_,
	      TVector2 &&met_,
	      bool has_bad_jets_,
	      int run_number_,
	      int event_number_,
	      double weight_,
	      double met_filter_,
	      bool HLT_xe70_)
		:
		leptons(leptons_),
		jets(jets_),
		bjets_60(bjets_60_),
		bjets_70(bjets_70_),
		bjets_85(bjets_85_),
		veryloose_tops(veryloose_tops_),
		loose_tops(loose_tops_),
		tight_tops(tight_tops_),
		smoothloose_tops(smoothloose_tops_),
		smoothtight_tops(smoothtight_tops_),
		met(met_),
		has_bad_jets(has_bad_jets_),
		run_number(run_number_),
		event_number(event_number_),
		weight(weight_),
		loose_tops_b60(get_top_b_match(loose_tops, bjets_60)),
		tight_tops_b60(get_top_b_match(tight_tops, bjets_60)),
		loose_tops_b70(get_top_b_match(loose_tops, bjets_70)),
		tight_tops_b70(get_top_b_match(tight_tops, bjets_70)),
		loose_tops_b85(get_top_b_match(loose_tops, bjets_85)),
		tight_tops_b85(get_top_b_match(tight_tops, bjets_85)),
		met_filter(met_filter_),
		HLT_xe70(HLT_xe70_)
		{};

	vector<TLorentzVector> leptons;
	vector<TLorentzVector> jets;
	vector<TLorentzVector> bjets_60;
	vector<TLorentzVector> bjets_70;
	vector<TLorentzVector> bjets_85;
	vector<TLorentzVector> veryloose_tops;
	vector<TLorentzVector> loose_tops;
	vector<TLorentzVector> tight_tops;
	vector<TLorentzVector> smoothloose_tops;
	vector<TLorentzVector> smoothtight_tops;

	TVector2 met;
	bool has_bad_jets;
	int run_number;
	int event_number;
	double weight;
	vector<TLorentzVector> loose_tops_b60;
	vector<TLorentzVector> tight_tops_b60;
	vector<TLorentzVector> loose_tops_b70;
	vector<TLorentzVector> tight_tops_b70;
	vector<TLorentzVector> loose_tops_b85;
	vector<TLorentzVector> tight_tops_b85;
	double met_filter;
	bool HLT_xe70;
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

vector<TLorentzVector> get_jets(InData &data)
{
	vector<TLorentzVector> jets;

	for (size_t i = 0; i < data.jets_pt->size(); i++) {
		double pt = data.jets_pt->at(i);
		double eta = data.jets_eta->at(i);
		double phi = data.jets_phi->at(i);
		double e = data.jets_e->at(i);
		if (pt > 30 && abs(eta) < 2.8)
			jets.push_back(make_tlv(pt,eta,phi,e));
	}

	sort(jets.begin(), jets.end(), compare_tlv);

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

bool has_bad_jets(InData &data)
{
	for (size_t i = 0; i < data.jets_pt->size(); ++i) {
		if (data.jets_pt->at(i) > 20 && data.jets_isBad->at(i))
			return true;
	}
	return false;
}

Event get_event(InData& data)
{
	double weight =
		data.weight_mc *
		data.weight_btag *
		data.weight_elec *
		data.weight_muon;
		//data.weight_pu;

	Event evt(get_leptons(data),
		  get_jets(data),
		  // FIXME
		  vector<TLorentzVector>(),
		  get_bjets(data), // OP = 77%
		  // FIXME
		  vector<TLorentzVector>(),
		  // FIXME
		  vector<TLorentzVector>(),
		  vector<TLorentzVector>(),
		  vector<TLorentzVector>(),
		  vector<TLorentzVector>(),
		  vector<TLorentzVector>(),
		  get_met(data),
		  // this crashes the v13 ntuples
		  //has_bad_jets(data),
		  false,
		  data.run_number,
		  data.event_number,
		  weight,
		  data.gen_filt_met,
		  data.trigger->at(2));
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

void fill_outdata(Event &evt, OutData &outdata, double scale)
{
	outdata.meff = calc_meff(evt.jets,evt.leptons,evt.met);
	outdata.met = evt.met.Mod();
	outdata.mt = calc_mt(evt.leptons, evt.met);
	outdata.jet_1_pt = PT_AT(0,evt.jets);
	outdata.jet_4_pt = PT_AT(3,evt.jets);
	outdata.jet_5_pt = PT_AT(4,evt.jets);
	outdata.jet_6_pt = PT_AT(5,evt.jets);
	outdata.jet_7_pt = PT_AT(6,evt.jets);
	outdata.jet_8_pt = PT_AT(7,evt.jets);
	outdata.n_b_60 = evt.bjets_60.size();
	outdata.n_b_70 = evt.bjets_70.size();
	outdata.n_b_85 = evt.bjets_85.size();
	outdata.mt_min_b_60 = calc_mt_min_bjets(evt.bjets_60, evt.met);
	outdata.mt_min_b_70 = calc_mt_min_bjets(evt.bjets_70, evt.met);
	outdata.mt_min_b_85 = calc_mt_min_bjets(evt.bjets_85, evt.met);
	outdata.veryloose_top_1_pt = PT_AT(0,evt.veryloose_tops);
	outdata.veryloose_top_2_pt = PT_AT(1,evt.veryloose_tops);
	outdata.loose_top_1_pt = PT_AT(0,evt.loose_tops);
	outdata.loose_top_2_pt = PT_AT(1,evt.loose_tops);
	outdata.tight_top_1_pt = PT_AT(0,evt.tight_tops);
	outdata.tight_top_2_pt = PT_AT(1,evt.tight_tops);
	outdata.smoothloose_top_1_pt = PT_AT(0,evt.smoothloose_tops);
	outdata.smoothloose_top_2_pt = PT_AT(1,evt.smoothloose_tops);
	outdata.smoothtight_top_1_pt = PT_AT(0,evt.smoothtight_tops);
	outdata.smoothtight_top_2_pt = PT_AT(1,evt.smoothtight_tops);
	outdata.loose_top_b60_1_pt = PT_AT(0, evt.loose_tops_b60);
	outdata.loose_top_b60_2_pt = PT_AT(1, evt.loose_tops_b60);
	outdata.tight_top_b60_1_pt = PT_AT(0, evt.loose_tops_b60);
	outdata.tight_top_b60_2_pt = PT_AT(1, evt.loose_tops_b60);
	outdata.loose_top_b70_1_pt = PT_AT(0, evt.loose_tops_b70);
	outdata.loose_top_b70_2_pt = PT_AT(1, evt.loose_tops_b70);
	outdata.tight_top_b70_1_pt = PT_AT(0, evt.loose_tops_b70);
	outdata.tight_top_b70_2_pt = PT_AT(1, evt.loose_tops_b70);
	outdata.loose_top_b85_1_pt = PT_AT(0, evt.loose_tops_b85);
	outdata.loose_top_b85_2_pt = PT_AT(1, evt.loose_tops_b85);
	outdata.tight_top_b85_1_pt = PT_AT(0, evt.loose_tops_b85);
	outdata.tight_top_b85_2_pt = PT_AT(1, evt.loose_tops_b85);
	outdata.event_number = evt.event_number;
	outdata.run_number = evt.run_number;
	outdata.weight = evt.weight * scale;
	outdata.gen_filt_met = evt.met_filter;
	outdata.n_lepton = evt.leptons.size();

	outdata.jets_pt = new vector<double>;
	for (TLorentzVector j : evt.jets)
		outdata.jets_pt->push_back(j.Pt());
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

bool good_event(Event &event, bool met_filter_under200, bool met_filter_over200)
{
	return
		   (!met_filter_under200 || event.met_filter < 200)
		&& (!met_filter_over200  || event.met_filter >= 200)
		&& (event.HLT_xe70)
		// this crashes the v13 ntuples
		//&& (!event.has_bad_jets)
		&& (event.jets.size() >= 4)
		&& (event.bjets_85.size() >= 2)
		&& (event.leptons.size() >= 1);
}

int main(int argc, char *argv[])
{
	bool met_filter_over200 = false;
	bool met_filter_under200 = false;
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
		met_filter_over200 = argv[1][1] == 'F';
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
		if (good_event(evt, met_filter_under200, met_filter_over200)) {
			fill_outdata(evt,outdata,scale);
			outtree.Fill();
			delete outdata.jets_pt;
		}
	}

	outfile.Write(0,TObject::kWriteDelete);
}
