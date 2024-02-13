// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// stcm7 runs a hippocampus model (hip_bench) with new ideas about temporal context
package main

import ( //import packages
	"bytes"
	"encoding/binary" //JWA
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand" //JWA
	"os"        //JWA

	//JWA
	"strconv"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/metric"   //JWA
	"github.com/emer/etable/minmax"   //JWA
	"github.com/emer/etable/simat"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/hip"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32" //JWA, had but removed "github.com/jinzhu/copier"
)

func main() { //main function that creates simulation
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
	//JWA changed below
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(dir)
	//JWA changed above
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// see def_params.go for the default params, and params.go for user-saved versions
// from the gui.

// see bottom of file for multi-factor testing params

// HipParams have the hippocampus size and connectivity parameters
type HipParams struct {
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
}

func (hp *HipParams) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}

// PatParams have the pattern parameters
type PatParams struct {
	ListSize    int     `desc:"number of A-B, A-C patterns each"`
	MinDiffPct  float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
	DriftCtxt   bool    `desc:"use drifting context representations -- otherwise does bit flips from prototype"`
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`
	DriftPct    float32 `desc:"percentage of active bits that drift, per step, for drifting context"`
}

//JWA, previously, before 6/12/21
type ErrLrateModParams struct {
	Base  float32    `min:"0" max:"1" desc:"baseline learning rate"`
	Err   float32    `desc:"multiplier on error factor"`
	Range minmax.F32 `viewif:"On" desc:"defines the range over which modulation occurs for the modulator factor -- Min and below get the Base level of learning rate modulation, Max and above get a modulation of 1"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net        *leabra.Network   `view:"no-inline"`
	Hip        HipParams         `desc:"hippocampus sizing parameters"`
	Pat        PatParams         `desc:"parameters for the input patterns"`
	ErrLrMod   ErrLrateModParams `desc:"parameters for the error lrn modulation"` //JWA
	PoolVocab  patgen.Vocab      `view:"no-inline" desc:"pool patterns vocabulary"`
	TrainAB    *etable.Table     `view:"no-inline" desc:"AB training patterns to use"`
	TrainAB2   *etable.Table     `view:"no-inline" desc:"AB training patterns to use"`
	TrainAB3   *etable.Table     `view:"no-inline" desc:"AB training patterns to use"`
	TrainAB4   *etable.Table     `view:"no-inline" desc:"AB training patterns to use"`
	TrainAB5   *etable.Table     `view:"no-inline" desc:"AB training patterns to use"`
	TrainAB6   *etable.Table     `view:"no-inline" desc:"AB training patterns to use"`
	TrainABnc  *etable.Table     `view:"no-inline" desc:"AB training patterns, no temporal context"`
	TrainAC    *etable.Table     `view:"no-inline" desc:"AC training patterns to use"`
	TrainAC2   *etable.Table     `view:"no-inline" desc:"AC training patterns to use"`
	TestAB     *etable.Table     `view:"no-inline" desc:"AB testing patterns to use"`
	TestABnc   *etable.Table     `view:"no-inline" desc:"AB testing patterns to use, no temp context"`
	TestAC     *etable.Table     `view:"no-inline" desc:"AC testing patterns to use"`
	TestACnc   *etable.Table     `view:"no-inline" desc:"AC testing patterns to use, no temp context"`
	TestLure   *etable.Table     `view:"no-inline" desc:"Lure testing patterns to use, AB list"`
	TestLurenc *etable.Table     `view:"no-inline" desc:"Lure testing patterns to use, no temp context"`
	//TestLureAC   *etable.Table            `view:"no-inline" desc:"Lure testing patterns to use, AC list"` //JWA
	TrainAll         *etable.Table            `view:"no-inline" desc:"all training patterns -- for pretrain"`
	TrnTrlLog        *etable.Table            `view:"no-inline" desc:"training trial-level log data"`
	TrnTrlLogLst     *etable.Table            `view:"no-inline" desc:"training trial-level log data, last trial"`
	TrnTrlLogItemLst *etable.Table            `view:"no-inline" desc:"training trial-level log data, last trial"`
	TrnEpcLog        *etable.Table            `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog        *etable.Table            `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog        *etable.Table            `view:"no-inline" desc:"testing trial-level log data"`
	TstCycLog        *etable.Table            `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog           *etable.Table            `view:"no-inline" desc:"summary log of each run"`
	RunStats         *etable.Table            `view:"no-inline" desc:"aggregate stats on all runs"`
	TstStats         *etable.Table            `view:"no-inline" desc:"testing stats"`
	TrainSimMats     *etable.Table            `view:"simmats for printing during training"`
	SimMats          map[string]*simat.SimMat `view:"no-inline" desc:"similarity matrix results for layers"`
	SimMatsQ2        map[string]*simat.SimMat `view:"no-inline" desc:"similarity matrix results for layers, Q2"`
	SimMatsLst       map[string]*simat.SimMat `view:"no-inline" desc:"similarity matrix results for layers last trial"`
	SimMatsItemLst   map[string]*simat.SimMat `view:"no-inline" desc:"similarity matrix results for layers last trial"`
	Params           params.Sets              `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string                   `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag              string                   `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params)"`
	MaxRuns          int                      `desc:"maximum number of model runs to perform"`
	MaxEpcs          int                      `desc:"maximum number of epochs to run per model run"`
	Cycs             int                      `desc:"# of alpha cycles / epoch"`                               //JWA
	CycsMultC        int                      `desc:"When multiple cycles, do we give no context or context?"` //JWA
	TCycs            int                      `desc:"# of alpha cycles during test"`                           //JWA
	driftbetween     int                      `desc:"drift between epochs of training (1) or not (0)"`         //JWA
	PreTrainEpcs     int                      `desc:"number of epochs to run for pretraining"`
	NZeroStop        int                      `desc:"if a positive number, training will stop after this many epochs with zero mem errors"`
	TrainEnv         env.FixedTable           `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv          env.FixedTable           `desc:"Testing environment -- manages iterating over testing"`
	Time             leabra.Time              `desc:"leabra timing parameters and state"`
	ViewOn           bool                     `desc:"whether to update the network view while running"`
	TrainUpdt        leabra.TimeScales        `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt         leabra.TimeScales        `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval     int                      `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	MemThr           float64                  `desc:"threshold to use for memory test -- if error proportion is below this number, it is scored as a correct trial"`
	pfix             string                   `desc:"prefix for inputs"` //JWA added from here on!
	tsoff            string                   `desc:"timescale offset to slow drift"`
	edl              int                      `desc:"error-driven learning on/off"`
	edlval           int                      `desc:"error-driven learning on/off val for toggling"`
	expnum           int                      `desc:"experiment number"`
	runnum           int                      `desc:"counts up current run number"`
	nints            int                      `desc:"# of intervals / exp type"`
	drifttype        int                      `desc:"drift type on current run"`
	drifttypes       int                      `desc:"# of types of drift we will run"`
	wpvc             int                      `desc:"# word pair vector columns"`
	lvc              int                      `desc:"# list vector columns"`
	cvcn             int                      `desc:"# context vector columns as int"`
	do_sequences     int                      `desc:"sequences and temporal community structure"`
	seq_numact       int                      `desc:"# activated in seq/tcs data"`
	seq_tce          int                      `desc:"temporal context on or not during seq/tcs encoding"`
	seq_dcurr        int                      `desc:"decrease activation for current relative to future item"`
	exptype          int                      `desc:"experiment type (e.g., fcurve)"`
	interval         int                      `desc:"retention interval"`
	targortemp       int                      `desc:"test the target (1, default) or temporal context (2)"`
	ECtoDGnl         int                      `desc:"no learning from EC to DG"`
	ECtoCA3nl        int                      `desc:"no learning from EC to CA3"`
	ECtoCA1nl        int                      `desc:"no learning from EC to CA1"`
	CA3toCA1nl       int                      `desc:"no learning from CA3 to CA1"`
	CA3toCA3nl       int                      `desc:"no learning from CA3 to CA1"`
	lratemulton      int                      `desc:"turn on / off lrate multiplier"`
	spect_type       int                      `desc:"type of drift in temp pools"`
	synap_decay      int                      `desc:"implement synaptic decay?"`
	decay_rate       float64                  `desc:"decay rate (if decay on)"`
	testlag          int                      `desc:"store testlag?"`
	fillers          [5]int                   `desc:"fscale values"`
	smithetal        int                      `desc:"smith et al decontextualization experiment if non-zero"`
	ttrav            int                      `desc:"mental time travel experiments"`
	ece_start        int                      `desc:"expanding/contrasting/equal exp start num"`
	rawson_start     int                      `desc:"rawson exp start num"`
	cepeda_start     int                      `desc:"cepeda exp start num"`
	cepeda_stop      int                      `desc:"cepeda exp stop num"`

	// statistics note: use float64 as that is best for etable.Table
	TestNm         string  `inactive:"+" desc:"what set of patterns are we currently testing"`
	Mem            float64 `inactive:"+" desc:"whether current trial's ECout met memory criterion"`
	TrgOnWasOffAll float64 `inactive:"+" desc:"current trial's proportion of bits where target = on but ECout was off ( < 0.5), for all bits"`
	TrgOnWasOffCmp float64 `inactive:"+" desc:"current trial's proportion of bits where target = on but ECout was off ( < 0.5), for only completion bits that were not active in ECin"`
	TrgOffWasOn    float64 `inactive:"+" desc:"current trial's proportion of bits where target = off but ECout was on ( > 0.5)"`
	TrlSSE         float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE      float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff     float64 `inactive:"+" desc:"current trial's cosine difference"`
	TrlEDCA3       float32 `inactive:"+" desc:"use this to modify specifically CA3 in LRateMult"` //JWA added from here on!
	DGED14         float64 `inactive:"+" desc:"err diffs"`
	CA3ED14        float64 `inactive:"+" desc:"err diffs"`
	CA1ED14        float64 `inactive:"+" desc:"err diffs"`
	ECoutED34      float64 `inactive:"+" desc:"err diffs"`

	EpcSSE        float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor     float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when Mem err first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero Mem err"`

	// internal state - view:"-"
	SumSSE       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	CntErr       int                         `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnTrlPlot   *eplot.Plot2D               `view:"-" desc:"the training trial plot"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D               `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	RunStatsPlot *eplot.Plot2D               `view:"-" desc:"the run stats plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	TrnEpcHdrs   bool                        `view:"-" desc:"headers written"`
	TstEpcFile   *os.File                    `view:"-" desc:"log file"`
	TstEpcHdrs   bool                        `view:"-" desc:"headers written"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	TmpVals      []float32                   `view:"-" desc:"temp slice for holding values -- prevent mem allocs"`
	TmpValsDWt   []float32                   `view:"-" desc:"temp slice for holding dwt values -- prevent mem allocs"`                //JWA
	TmpValsWtR   []float32                   `view:"-" desc:"temp slice for holding wt values from rec to ca3 -- prevent mem allocs"` //JWA
	LayStatNms   []string                    `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	TstNms       []string                    `view:"-" desc:"names of test tables"`
	SimMatStats  []string                    `view:"-" desc:"names of sim mat stats"`
	TstStatNms   []string                    `view:"-" desc:"names of test stats"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	PreTrainWts  []byte                      `view:"-" desc:"pretrained weights file"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64                       `view:"-" desc:"the current random seed"`
	LastEpcTime  time.Time                   `view:"-" desc:"timer for last epoch"`
}

//JWA, add if you want global learning rate change based on error (not used in spacing effect paper)
func (em *ErrLrateModParams) Defaults() {
	em.Base = 0.01           //JWA, was 0.1
	em.Err = 1               //JWA, was 4
	em.Range.Set(0.65, 1.02) //try 7/1/21 to stretch out the space, 0.75-1.05
}
func (em *ErrLrateModParams) Update() {
}

// LrateMod returns the learning rate modulation as a function of any kind of normalized error measure
func (em *ErrLrateModParams) LrateMod(err float32) float32 {
	lrm := float32(1)
	switch {
	case err < em.Range.Min:
		lrm = 0
	case err > em.Range.Max:
		lrm = 1
	default:
		lrm = em.Range.NormVal(err)
	}
	mod := em.Base + lrm*(1-em.Base)
	return mod
}

//end add from before 6/12/21

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &etable.Table{}
	ss.TrainAB2 = &etable.Table{}
	ss.TrainAB3 = &etable.Table{}
	ss.TrainAB4 = &etable.Table{}
	ss.TrainAB5 = &etable.Table{}
	ss.TrainAB6 = &etable.Table{}
	ss.TrainABnc = &etable.Table{}
	ss.TrainAC = &etable.Table{}
	ss.TrainAC2 = &etable.Table{}
	ss.TestAB = &etable.Table{}
	ss.TestABnc = &etable.Table{}
	ss.TestAC = &etable.Table{}
	ss.TestACnc = &etable.Table{}
	ss.TestLure = &etable.Table{}
	ss.TestLurenc = &etable.Table{}
	//ss.TestLureAC = &etable.Table{}
	ss.TrainAll = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnTrlLogLst = &etable.Table{}
	ss.TrnTrlLogItemLst = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.SimMats = make(map[string]*simat.SimMat)
	ss.SimMatsQ2 = make(map[string]*simat.SimMat)
	ss.SimMatsLst = make(map[string]*simat.SimMat)
	ss.SimMatsItemLst = make(map[string]*simat.SimMat)
	ss.Params = ParamSets // in def_params -- current best params
	// ss.Params = OrigParamSets // original, previous model
	// ss.Params = SavedParamsSets // current user-saved gui params
	ss.RndSeed = 2
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.TestInterval = 1
	ss.targortemp = 1 //JWA, test target or temporal context reinstatement?
	ss.LogSetParams = false
	ss.MemThr = 0.2     //JWA, was 0.3
	ss.tsoff = "0"      //timescale offset to slow down drift - higher, slower drift
	ss.cvcn = 4         //context vector columns (note: 2 pools per column)
	ss.MaxEpcs = 5
	ss.lvc = 0          //list vector columns - implemented in smithetal exps - if you change lvc and not others, must change ECSize and the input patterns!
	ss.wpvc = 4         //word vector columns
	ss.TCycs = 1        //JWA, # of cycles to run during test
	ss.driftbetween = 1 //JWA, do we want to drift between epochs of training?

	//no learning manipulations
	ss.ECtoDGnl = 0        //JWA, no learning EC to DG #keep 0 normally
	ss.ECtoCA3nl = 0       //JWA, no learning EC to CA3 #keep 0 normally
	ss.ECtoCA1nl = 0       //JWA, no learning ECin to CA1 #keep 0 normally
	ss.CA3toCA1nl = 0      //JWA, no learning CA3 to CA1 - catastrophic effect #keep 0 normally
	ss.CA3toCA3nl = 0      //JWA, no learning CA3 to CA3 recurrent collaterals #keep 0 normally
	ss.edlval = 1          //JWA, switches EDL on/off through training #keep 1 normally, but test w/ 0
	ss.edl = ss.edlval + 0 //JWA, separate these so we can declare all up here

	ss.lratemulton = 0 //JWA, 1=lratemulton, uses lratemod, 0=not on

	ss.spect_type = 1      //JWA, 1=spectra (normal), 2=all fast, 3=all med, 4=all slow, 5=spectrum base change
	ss.synap_decay = 0     //JWA, turn on decay or no; not used
	ss.decay_rate = 0.0005 //JWA, decay rate, keep at 0.0005
	ss.smithetal = 0       //default, adjust later if this is a decontextualization experiment
	ss.ttrav = 0           // allow for timetravel during both re-learning and testing

	// these correspond to starting/stopping #s for different experiments. used to re-direct everything based on inputs from the meta script.
	//note that this influences expnum but expnum is much vaster, as it relates to every single retention interval too
	//ece = expanding/contracting/equal; rawson and cepeda experiments correspond to those in the paper.
	ss.ece_start, ss.rawson_start, ss.cepeda_start, ss.cepeda_stop = 11, 15, 18, 27

	ss.nints = 9                       //# of retention intervals; 0=no lag, 1-7 increasing RIs, 8=scramble
	ss.drifttypes = ss.cepeda_stop + 0 //experimental conditions
	ss.runnum = 0
	var nogui bool // JWA 2_18_21
	// DUPLICATE these flags up here so they can be used to read in file (otherwise it comes after read-in...)
	if len(os.Args) > 1 {
		flag.IntVar(&ss.expnum, "expnum", -1, "which specific experiment # to run")
		flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
		flag.IntVar(&ss.MaxRuns, "runs", 2, "number of runs to do")
		flag.IntVar(&ss.MaxEpcs, "epcs", 5, "number of epcs to do")
		flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
		flag.Parse()
	}
	//ss.expnum = 126        //uncomment this to impose an expnum while debugging - ALWAYS COMMENT OUT WHEN SUBMITTING ON CLUSTER
	if ss.expnum == -1 { // relevant only if not pre-set in bash code
		ss.expnum = 0 //
	}
	ss.do_sequences = 0 //test reviewer idea about sequences (1) or community structure (2). also use (3) for drift only to measure drift across layers
	ss.seq_numact = 0 //if #activated units == 2 (ss.seq_dcurr)
	ss.seq_tce = 0 //is temporal context ON or not at encoding (ss.seq_tce)
	ss.seq_dcurr = 0 //if we want to decrease activation for current item relative to future one? (1) or keep same (0)

	smiththresh:=500
	seqthresh:=550 //from 551-562
	travthresh:=600
	ttrav2thresh:=700
	ttrav3thresh:=800
	if ss.expnum > smiththresh && ss.expnum < seqthresh { // run smith et al decontextualization experiments
		ss.smithetal = ss.expnum - smiththresh //so 501 = 1, etc.
		ss.lvc = 1                     //list vector columns
		ss.cvcn -= ss.lvc              //context vector columns, adjust for lvc
		texpnum := 14                  //sample expnum to control for time (was 13 in initial draft, increased for better SNR)
		ss.exptype, ss.interval, ss.drifttype = texpnum/(ss.nints*ss.drifttypes), texpnum%ss.nints, texpnum/ss.nints
	} else if ss.expnum > seqthresh && ss.expnum < travthresh { //sequences
		buff := ss.expnum - (seqthresh+1) //use original values after this change in input
		texpnum := 5
		ss.exptype, ss.interval, ss.drifttype = texpnum/(ss.nints*ss.drifttypes), texpnum%ss.nints, texpnum/ss.nints
		ss.do_sequences=(buff/4)%3+1 //1, 2, or 3
		ss.seq_numact=(buff/2)%2+1 //1 or 2
		ss.seq_tce=buff%2 //0 or 1	
		fmt.Printf("do_sequences: %d\n", ss.do_sequences)
		fmt.Printf("seq_numact: %d\n", ss.seq_numact)
		fmt.Printf("seq_tce: %d\n", ss.seq_tce)
	} else if ss.expnum > travthresh && ss.expnum < ttrav2thresh { //ttrav with no context
		buff := ss.expnum - travthresh //use original values after this change in input
		ss.ttrav = 1
		ss.exptype, ss.interval, ss.drifttype = buff/(ss.nints*ss.drifttypes), buff%ss.nints, buff/ss.nints
	} else if ss.expnum > ttrav2thresh && ss.expnum < ttrav3thresh { //simply another learning cycle w/ same input
		buff := ss.expnum - ttrav2thresh
		ss.ttrav = 2
		ss.exptype, ss.interval, ss.drifttype = buff/(ss.nints*ss.drifttypes), buff%ss.nints, buff/ss.nints
	} else if ss.expnum > ttrav3thresh { //only ONE learning cycle, NO imposed temporal context
		buff := ss.expnum - ttrav3thresh
		ss.ttrav = 3
		ss.exptype, ss.interval, ss.drifttype = buff/(ss.nints*ss.drifttypes), buff%ss.nints, buff/ss.nints
	} else {
		ss.exptype, ss.interval, ss.drifttype = ss.expnum/(ss.nints*ss.drifttypes), ss.expnum%ss.nints, ss.expnum/ss.nints
	}
	
	if ss.do_sequences > 0 {
		ss.MaxEpcs=1 //only one long, long epoch
	}
	
	if ss.expnum >= ss.nints*ss.drifttypes && ss.expnum < smiththresh { //just go in order above a certain number
		ss.drifttype = ss.drifttypes + ss.expnum - ss.nints*ss.drifttypes
		ss.exptype = 0
	}

	if ss.exptype == 0 {
		ss.pfix = "fcurve/"
	} else if ss.exptype == 1 {
		ss.pfix = "sp/"
	} else if ss.exptype == 2 {
		ss.pfix = "pi/"
	} else if ss.exptype == 3 {
		ss.pfix = "ri/"
	} else if ss.exptype == 4 {
		ss.pfix = "rin/"
	}
	fmt.Printf("expnum: %d\n", ss.expnum)
	fmt.Printf("exptype: %d\n", ss.exptype)
	fmt.Printf("pfix: %s\n", ss.pfix)
	fmt.Printf("interval: %d\n", ss.interval)
	fmt.Printf("drift type: %d\n", ss.drifttype)
	fmt.Printf("ttrav: %d\n", ss.ttrav)
	ss.LayStatNms = []string{"ECin", "ECout", "DG", "CA3", "CA1"}
	ss.TstNms = []string{"AB", "AC", "Lure"}
	ss.TstStatNms = []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}
	ss.SimMatStats = []string{"Within", "Between"}
	ss.Defaults()
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 16    // # paired associates
	pp.MinDiffPct = 0.5 //JWA - keep at 0.5 for most things
	pp.CtxtFlipPct = .02
	pp.DriftPct = 0.0005 //JWA: this is from a prior hip_bench and I control drift more elaborately below!
}

func (hp *HipParams) Defaults() {
	// size
	hp.ECSize.Set(2, 8) // JWA, if you change this, you will need to change Gi in def_params file up or down to accommodate change in necessary inhibition

	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(20, 20) //JWA, BigHip is 20, MedHip is 15, SmallHip 10
	hp.CA3Size.Set(40, 40) //JWA, BigHip is 40, MedHip is 30, SmallHip 20
	hp.DGRatio = 1.5

	// ratio
	hp.DGPCon = 0.25 // .35 is sig worse, .2 learns faster but AB recall is worse
	hp.CA3PCon = 0.25
	hp.MossyPCon = 0.02 // .02 > .05 > .01 (for small net)
	hp.ECPctAct = 0.2   //0.2

	hp.MossyDel = 4     // 4 > 2 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 3 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount..
}

func (ss *Sim) Defaults() {
	ss.Hip.Defaults()
	ss.Pat.Defaults()
	ss.ErrLrMod.Defaults() //JWA
	ss.Time.CycPerQtr = 25 // note: key param - 25 seems like it is actually fine?
	ss.Update()
}

func (ss *Sim) Update() {
	ss.Hip.Update()
	ss.ErrLrMod.Update() //JWA
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	pass := 0
	if ss.MaxRuns == 0 { // allow user override in GUI or scripts
		ss.MaxRuns = 40 //20 JWA 5_8_23 was 20
		pass = 1
	}

	if ss.NoGui {
		fmt.Printf("noguiflag!\n")
		pass = 1
	}
	if pass == 1 {
		ss.Cycs = 1      //JWA, # of alpha cycles to run / epoch (default = 1)
		ss.CycsMultC = 1 //JWA, if Cycs>1, give context (1) (default) or no (0)
		if ss.ttrav == 1 {
			ss.CycsMultC = 0 //if ttrav, impose 0
		}
		ss.NZeroStop = 1
		ss.PreTrainEpcs = 6 // seems sufficient? //JWA, was 8
	}
	fmt.Printf("maxepcs: %d\n", ss.MaxEpcs)
	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB) //JWA, this is first trained set
	ss.TrainEnv.Sequential = true                     //JWA add - train in order!!
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestAB)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

// SetEnv select which set of patterns to train on: AB or AC
func (ss *Sim) SetEnv(trainAC bool) {
	if trainAC {
		ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAC)
	} else {
		ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	}
	ss.TrainEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "stcm7") //JWA, was "Hip_bench"
	hp := &ss.Hip
	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Input)
	ecin := net.AddLayer4D("ECin", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Hidden)
	ecout := net.AddLayer4D("ECout", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Target) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, emer.Hidden)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, emer.Hidden)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	ecin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	ecout.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ECin", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(in, ecin, onetoone, emer.Forward)
	net.ConnectLayers(ecout, ecin, onetoone, emer.Back)

	// EC <-> CA1 encoder pathways
	pj := net.ConnectLayersPrjn(ecin, ca1, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ca1, ecout, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ecout, ca1, pool1to1, emer.Back, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")

	// Perforant pathway
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon

	pj = net.ConnectLayersPrjn(ecin, dg, ppathDG, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	if true { // toggle for bcm vs. ppath
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
	} else {
		// so far, this is sig worse, even with error-driven MinusQ1 case (which is better than off)
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL") //JWA, from Alan, was "PPath"
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL") //JWA, from Alan, was "PPath"
	}

	// always use this for now:
	if true {
		pj = net.ConnectLayersPrjn(ca3, ca1, full, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
	} else {
		// note: this requires lrate = 1.0 or maybe 1.2, doesn't work *nearly* as well
		pj = net.ConnectLayers(ca3, ca1, full, emer.Forward) // default con
		// pj.SetClass("HippoCHL")
	}

	// Mossy fibers
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hp.MossyPCon
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, &hip.CHLPrjn{}) // no learning
	pj.SetClass("HippoCHL")

	// using 4 threads total (rest on 0)
	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3) // this has the most

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

func (ss *Sim) ReConfigNet() {
	//JWA
	ss.Update() //JWA, based on Alan's comment
	ss.ConfigPats()
	ss.Net = &leabra.Network{} // start over with new network
	ss.ConfigNet(ss.Net)
	if ss.NetView != nil {
		ss.NetView.SetNet(ss.Net)
		ss.NetView.Update() // issue #41 closed
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.ReConfigNet()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.NewRun()
	ss.UpdateView(true)
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	/*if train {
		ss.Net.WtFmDWt()
	}*/ //4/29/22 copied down below to fix DWt rounding issue

	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra() //JWA
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(leabra.LeabraPrjn).AsLeabra()
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()
	ca3FmCa3 := ca3.RcvPrjns.SendName("CA3").(leabra.LeabraPrjn).AsLeabra()
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra() //JWA add
	DgFmECin := dg.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra()
	_ = ecin
	_ = input
	if ss.ECtoDGnl == 1 {
		DgFmECin.Learn.Learn = false
	}
	if ss.ECtoCA3nl == 1 {
		ca3FmECin.Learn.Learn = false
	}
	if ss.ECtoCA1nl == 1 {
		ca1FmECin.Learn.Learn = false
	}
	if ss.CA3toCA1nl == 1 {
		ca1FmCa3.Learn.Learn = false
	}
	if ss.CA3toCA3nl == 1 {
		ca3FmCa3.Learn.Learn = false
	}

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0

	dgwtscale := ca3FmDg.WtScale.Rel
	// 0 for the first quarter, comment out if testing in orig, zycyc
	//JWA, comment out to turn off error-driven learning
	if ss.edl == 1 {
		ca3FmDg.WtScale.Rel = dgwtscale - ss.Hip.MossyDel
	}

	if train {
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		ecout.SetType(emer.Compare) // don't clamp
	}
	ecout.UpdateExtFlags() // call this after updating type

	ss.Net.AlphaCycInit() //JWA: triggers decay to run in most/all layers (e.g. CA3)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1

			//ca3FmDg.WtScale.Rel = dgwtscale //JWA
			if train {
				ca3FmDg.WtScale.Rel = dgwtscale
			} else {
				ca3FmDg.WtScale.Rel = dgwtscale - ss.Hip.MossyDelTest // testing
			}
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
			if train {              // clamp ECout from ECin
				ecin.UnitVals(&ss.TmpVals, "Act") // note: could use input instead -- not much diff
				ecout.ApplyExt1D32(ss.TmpVals)
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	ca3FmDg.WtScale.Rel = dgwtscale // restore
	ca1FmCa3.WtScale.Abs = 1
	ss.TrialStats(train) //JWA, moved from TrainTrial: logic here, this gives you error signal from this trial

	if train {
		//JWA, for "neuromodulation"
		//ss.ErrLrMod.LrateMod(float32(1 - ss.TrlCosDiff)) // previous
		if ss.lratemulton == 0 {
			ss.ErrLrMod.LrateMod(float32(ss.TrlEDCA3))
		} else {
			lrm := ss.ErrLrMod.LrateMod(float32(ss.TrlEDCA3)) //JWA
			ss.Net.LrateMult(lrm)                             //JWA, this multiplies lrate based on lrm
		}
		ss.Net.DWt()
		ss.Net.WtFmDWt() //4/29/22 fixed, added from above in AlphaCyc, not in original hip_bench here
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		if ss.TstCycPlot != nil {
			ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
		}
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	// JWA - this clears prior trial
	lays := []string{"Input", "ECout"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) ApplyInputsNC(en env.Env) { //apply inputs without clearing!
	lays := []string{"Input", "ECout"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun() //assigns ss.NeedsNewRun to false so ss.PreTrain() won't call it...
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		//JWA new code
		if ss.pfix == "fcurve/" { //for forgetting curve, no switch
			if ss.driftbetween > 0 {
				if epc == 0 {
					ss.edl = ss.edlval // use EDL for AB learning
				} else if epc == 1 { // JWA here it switches from AB->AB2
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB2)
				} else if epc == 2 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB3)
				} else if epc == 3 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB4)
				} else if epc == 4 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB5)
				} else if epc == 5 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB6)
				}
			} else { //if driftbetween=0, no switch to different AB patterns
				if epc == 0 {
					ss.edl = ss.edlval
				}
			}
			// JCD/Alan fix: set names after updating epochs to get correct names for the next env
			ss.TrainEnv.SetTrialName()
			ss.TrainEnv.SetGroupName()
		}
		if ss.pfix != "fcurve/" {
			if ss.driftbetween > 0 {
				if epc == 0 {
					ss.edl = ss.edlval
				} else if epc == 1 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB2)
				} else if epc == 2 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB3)
				} else if epc == 3 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB4)
				} else if epc == 4 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB5)
				} else if epc == 5 {
					ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB6)
				}
			} else {
				if epc == 0 {
					ss.edl = ss.edlval
				}
			}
		}
		if epc >= ss.MaxEpcs || ss.MaxEpcs == 0 { // done with training. //JWA added || part?
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.runnum += 1
				fmt.Printf("new runnum: %d\n", ss.runnum)
				ss.NeedsNewRun = true
				return
			}
		}
	}

	//ss.ApplyInputs(&ss.TrainEnv)
	//ss.AlphaCyc(true) // train //JWA
	//ss.TrialStats(true) // accumulate
	//ss.LogTrnTrl(ss.TrnTrlLog)
	solonc := 0 //0=default (imposted temporal context),1=solo cycle but no temporal context
	if ss.ttrav > 0 && epc > 0 {
		ss.Cycs = 2 //if time travel on, allow for 2nd cycle to think back
	}
	if ss.ttrav == 3 && epc > 0 {
		ss.Cycs = 1 //only one cycle, but...
		solonc = 1  //no temporal context
	}

	for i := 0; i < ss.Cycs; i++ {
		temptable := ss.TrainEnv.Table
		if i == 0 {
			if solonc == 0 {
				ss.Net.InitActs() //JWA, this keeps the decay ON
				ss.ApplyInputs(&ss.TrainEnv)
			} else {
				ss.Net.InitExt()
				ss.TrainEnv.Table = etable.NewIdxView(ss.TrainABnc)
				ss.ApplyInputsNC(&ss.TrainEnv)
			}
		} else { //JWA, this allows "mental time travel" by blanking out current temporal inputs - currently, with Cycs = 1, this is not being evaluated
			ss.Net.InitExt()       //JWA, clear out ApplyInputs
			if ss.CycsMultC == 0 { // give no context, allow time travel
				ss.TrainEnv.Table = etable.NewIdxView(ss.TrainABnc)
			} else if ss.CycsMultC == 1 { // impose training context
				ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
			}
			ss.ApplyInputsNC(&ss.TrainEnv)
		}
		ss.AlphaCyc(true) // train
		if ss.Cycs > 1 {
			if i == ss.Cycs-1 {
				ss.TrainEnv.Table = temptable //set back to orig table
			}
		}
	}

	if ss.ttrav > 0 && epc >= ss.MaxEpcs { //set back at end of run
		ss.Cycs = 1 //if time travel on, allow for 2nd cycle to think back
	}

	//ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

// PreTrainTrial runs one trial of pretraining using TrainEnv
func (ss *Sim) PreTrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.PreTrainEpcs { // done with training..
			ss.StopNow = true
			return
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.ConfigPats() //JWA, added, 4/7/21 //try changing 11/19/21
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.LrateMult(1) //JWA add, via Randy's zulip comment
	ss.Net.InitWts()
	ss.LoadPretrainedWts()
	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

func (ss *Sim) LoadPretrainedWts() bool {
	if ss.PreTrainWts == nil {
		fmt.Printf("no pretraining\n")
		return false
	}
	b := bytes.NewReader(ss.PreTrainWts)
	err := ss.Net.ReadWtsJSON(b)
	if err != nil {
		log.Println(err)
	} else {
		fmt.Printf("loaded pretrained wts\n")
	}
	return true
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.CntErr = 0
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.Mem = 0
	ss.TrgOnWasOffAll = 0
	ss.TrgOnWasOffCmp = 0
	ss.TrgOffWasOn = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Targ values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
func (ss *Sim) MemStats(train bool) {
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	//nn := ecout.Shape().Len() //JWA edit
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIdx("ActM")
	targi, _ := ecout.UnitVarIdx("Targ")
	actQ1i, _ := ecout.UnitVarIdx("ActQ1")
	hp := &ss.Hip
	ind1 := ss.wpvc * hp.ECPool.Y * hp.ECPool.X     //start after cue
	ind2 := ss.wpvc * 2 * hp.ECPool.Y * hp.ECPool.X //go until end of target (assumes cue and target sizes are equal)
	if ss.targortemp == 2 {                         // test temporal context instead
		ind1 = ss.wpvc * 2 * hp.ECPool.Y * hp.ECPool.X
		ind2 = (ss.wpvc + ss.cvcn) * 2 * hp.ECPool.Y * hp.ECPool.X
	}
	//fmt.Printf("\n")
	for ni := ind1; ni < ind2; ni++ {
		//for ni := 0; ni < nn; ni++ { //JWA this was old code that started from 0
		actm := ecout.UnitVal1D(actMi, ni)
		trg := ecout.UnitVal1D(targi, ni) // full pattern target
		inact := ecin.UnitVal1D(actQ1i, ni)
		/*if ni > ind2-(hp.ECPool.Y*hp.ECPool.X) { //print last pool?
			fmt.Printf("trg: %v\n", trg)
			fmt.Printf("actm: %v\n", actm)
			fmt.Printf("inact: %v\n", inact)
		}*/
		if trg < 0.5 { // trgOff
			trgOffN += 1
			if actm > 0.5 {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < 0.5 { // missing in ECin -- completion target
				cmpN += 1
				if actm < 0.5 {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < 0.5 {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	//fmt.Printf("trgOnN: %v\n", trgOnN)
	//fmt.Printf("cmpN: %v\n", cmpN)
	trgOnWasOffAll /= trgOnN //JWA, miss rate
	trgOffWasOn /= trgOffN   //JWA, false alarm rate
	if train {               // no cmp
		if trgOnWasOffAll < ss.MemThr && trgOffWasOn < ss.MemThr {
			ss.Mem = 1
		} else {
			ss.Mem = 0
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < ss.MemThr && trgOffWasOn < ss.MemThr {
				ss.Mem = 1
			} else {
				ss.Mem = 0
			}
		}
	}
	ss.TrgOnWasOffAll = trgOnWasOffAll
	ss.TrgOnWasOffCmp = trgOnWasOffCmp
	ss.TrgOffWasOn = trgOffWasOn
	//fmt.Printf("trgOnWasOffAll: %v\n", trgOnWasOffAll)
	//fmt.Printf("trgOnWasOffCmp: %v\n", trgOnWasOffCmp)
	//fmt.Printf("trgOffWasOn: %v\n", trgOffWasOn)
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats

// JWA, in minus phase vs plus phase of ECout, what's the difference?
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) { //JWA do i need to add avgsset here?
	outLay := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ss.TrlCosDiff = float64(outLay.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
		if ss.TrlSSE != 0 {
			ss.CntErr++
		}
	}
	// JWA, grab Q1 vs Q4 and do the error metric and create this
	// these are going to be pretty tiny - print them out! will need to look at this and THIS is what we're changing w/ LRateMult
	// call this in LRateMult instead of TrlCosDiff
	lnm := "CA3"
	ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
	tsrq1 := ss.ValsTsr(lnm + "Q1") //JWA, end of Q1 is before DG input, end of Q2 is after 1/4 DG input, end of Q3 is guess, end of Q4 is post +
	tsrq4 := ss.ValsTsr(lnm + "ActP")
	ly.UnitValsTensor(tsrq1, "ActQ1") // create tsr1 name, etc.
	ly.UnitValsTensor(tsrq4, "ActP")
	actavgg := ly.Pools[0].Inhib.Act.Avg //average for this single trial                     //JWA, effective, most stable
	ss.TrlEDCA3 = metric.Abs32(tsrq1.Values, tsrq4.Values) / (actavgg * float32(tsrq4.Len()))
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// SetDgCa3Off sets the DG and CA3 layers off (or on)
func (ss *Sim) SetDgCa3Off(net *leabra.Network, off bool) {
	ca3 := net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	dg := net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca3.Off = off
	dg.Off = off
}

// PreTrain runs pre-training, saves weights to PreTrainWts
func (ss *Sim) PreTrain() {
	ss.SetDgCa3Off(ss.Net, true)
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAll)
	ss.TrainEnv.Init(ss.TrainEnv.Run.Cur) //JWA, from XL!
	// todo: pretrain on all patterns!
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur //JWA
	for {
		ss.PreTrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	b := &bytes.Buffer{}
	ss.Net.WriteWtsJSON(b)
	ss.PreTrainWts = b.Bytes()
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TrainEnv.Init(ss.TrainEnv.Run.Cur) //JWA, from XL!
	ss.SetDgCa3Off(ss.Net, false)
	ss.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool, abaclure int) {
	ss.TestEnv.Step()

	// Query counters FIRST
	//_, _, chg := ss.TestEnv.Counter(env.Epoch)
	epc, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		if returnOnChg {
			return
		}
	}

	//changes 2_10_21
	//ss.ApplyInputs(&ss.TestEnv)
	//ss.AlphaCyc(false)             // !train //JWA, was this
	//was in the below
	solonc := 0
	if ss.ttrav > 0 {
		ss.TCycs = 2 //if time travel on, allow for 2nd cycle to think back
	}
	if ss.ttrav == 3 {
		ss.TCycs = 1 //only one cycle, but...
		solonc = 1   //no temporal context
	}
	for i := 0; i < ss.TCycs; i++ { //JWA added this instead
		if i == 0 {
			if solonc == 0 {
				ss.Net.InitActs()           //JWA, 3_10_21 artificially on this trial put decay=0; 3_16_21 commenting this out helps
				ss.ApplyInputs(&ss.TestEnv) //JWA, do NOT keep applying inputs ...?
			} else {
				ss.Net.InitExt() //JWA, clear out ApplyInputs
				ss.TestEnv.Table = etable.NewIdxView(ss.TestABnc)
				ss.ApplyInputsNC(&ss.TestEnv)
			}
		} else { //JWA, this allows "mental time travel" by blanking out current temporal inputs - currently, with TCycs = 1, this is not being evaluated
			ss.Net.InitExt() //JWA, clear out ApplyInputs
			if abaclure == 1 {
				ss.TestEnv.Table = etable.NewIdxView(ss.TestABnc)
			} else if abaclure == 2 {
				ss.TestEnv.Table = etable.NewIdxView(ss.TestACnc)
			} else if abaclure == 3 {
				ss.TestEnv.Table = etable.NewIdxView(ss.TestLurenc)
			}
			ss.ApplyInputsNC(&ss.TestEnv)
		}
		ss.AlphaCyc(false) // test
		if ss.TCycs > 1 {
			if i == ss.TCycs-1 {
				if abaclure == 1 {
					ss.TestEnv.Table = etable.NewIdxView(ss.TestAB) //set back
				} else if abaclure == 2 {
					ss.TestEnv.Table = etable.NewIdxView(ss.TestAC)
				} else if abaclure == 3 {
					ss.TestEnv.Table = etable.NewIdxView(ss.TestLure)
				}
			}
		}
	}
	if ss.ttrav > 0 && epc >= ss.MaxEpcs { //set back
		ss.Cycs = 1
	}
	//end of changes
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestNm = "AB"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestAB)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true, 1) // return on chg, abtest or no
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim", "Hip", "Pat"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}

	if sheet == "" || sheet == "Hip" {
		simp, ok := pset.Sheets["Hip"]
		if ok {
			simp.Apply(&ss.Hip, setMsg)
		}
	}

	if sheet == "" || sheet == "Pat" {
		simp, ok := pset.Sheets["Pat"]
		if ok {
			simp.Apply(&ss.Pat, setMsg)
		}
	}

	//JWA
	if sheet == "" || sheet == "ErrLrMod" {
		simp, ok := pset.Sheets["ErrLrMod"]
		if ok {
			simp.Apply(&ss.ErrLrMod, setMsg)
		}
	}

	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) OpenPat(dt *etable.Table, fname, name, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func (ss *Sim) ConfigPats() {
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y       // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X       // makes much more readable
	npats := ss.Pat.ListSize //4
	ece_start, rawson_start, cepeda_start, cepeda_stop := ss.ece_start, ss.rawson_start, ss.cepeda_start, ss.cepeda_stop
	pctAct := hp.ECPctAct
	minDiff := ss.Pat.MinDiffPct
	seqver := 2   //1=15-state version, 2=8-state version
	nstates := 15 // for temporal community simulations only
	ntrans := nstates * 4
	if seqver == 2 {
		nstates = 8
		ntrans = nstates + 0
	}
	if ss.do_sequences > 0 { // for temporal community simulations
		npats = 160   // length of total sequence //was 900
		minDiff = 0.2 //allow smaller differences
	}
	//pctDrift := ss.Pat.DriftPct //customizing drift separately in our model
	cvcn := ss.cvcn
	exptype := ss.exptype
	interval := ss.interval
	nints := ss.nints
	//////////////////////// ESTABLISH INITIAL PATTERNS /////////////////////////////////
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX) //to blank out targets or other reasons
	//jwa 4/5/23 experiment

	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A1", npats, plY, plX, pctAct, minDiff) //cues
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A2", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A3", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A4", npats, plY, plX, pctAct, minDiff)
	//NOTE "A1-4" will be altered in the RIn condition below
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B1", npats, plY, plX, pctAct, minDiff) //targets
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B2", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B3", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B4", npats, plY, plX, pctAct, minDiff)

	//list vectors - change below for Smith et al decontextualization experiments // these will be unused / overwritten if lvc=0
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
	patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_9", npats, "q", 0)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
	patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_10", npats, "q", 0)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
	patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_11", npats, "q", 0)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
	patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_12", npats, "q", 0)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_9", "ctxt_9") //same list context @ later learning, unless modified below because of spacing (most cases)
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_10", "ctxt_10")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_11", "ctxt_11")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_12", "ctxt_12")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_9", "ctxt_9")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_10", "ctxt_10")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_11", "ctxt_11")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_12", "ctxt_12")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_9", "ctxt_9")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_10", "ctxt_10")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_11", "ctxt_11")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_12", "ctxt_12")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_9", "ctxt_9")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_10", "ctxt_10")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_11", "ctxt_11")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_12", "ctxt_12")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_9", "ctxt_9")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_10", "ctxt_10")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_11", "ctxt_11")
	patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_12", "ctxt_12")
	patgen.AddVocabClone(ss.PoolVocab, "ctxtT_9", "ctxt_9") //same list context @ test, unless modified below because of a retention interval (most cases)
	patgen.AddVocabClone(ss.PoolVocab, "ctxtT_10", "ctxt_10")
	patgen.AddVocabClone(ss.PoolVocab, "ctxtT_11", "ctxt_11")
	patgen.AddVocabClone(ss.PoolVocab, "ctxtT_12", "ctxt_12")

	if ss.do_sequences > 0 { //sequences/temporal community structure
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "init", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb0", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb1", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb2", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb3", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb4", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb5", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb6", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gb7", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt0", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt1", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt2", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt3", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt4", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt5", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt6", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "gbt7", nstates, plY, plX, pctAct, minDiff)
		patgen.AddVocabRepeat(ss.PoolVocab, "z", 1, "empty", 0) //empty
		patgen.AddVocabRepeat(ss.PoolVocab, "a", 1, "init", 0)
		patgen.AddVocabRepeat(ss.PoolVocab, "b", 1, "init", 1)
		patgen.AddVocabRepeat(ss.PoolVocab, "c", 1, "init", 2)
		patgen.AddVocabRepeat(ss.PoolVocab, "d", 1, "init", 3)
		patgen.AddVocabRepeat(ss.PoolVocab, "e", 1, "init", 4)
		patgen.AddVocabRepeat(ss.PoolVocab, "f", 1, "init", 5)
		patgen.AddVocabRepeat(ss.PoolVocab, "g", 1, "init", 6)
		patgen.AddVocabRepeat(ss.PoolVocab, "h", 1, "init", 7)
		//import order from MATLAB script 'sims.m'
		//actual list is 600 trials

		//create same pattern as "a" but with lower activation (0.5)
		patgen.AddVocabClone(ss.PoolVocab, "smalla", "a")
		patgen.AddVocabClone(ss.PoolVocab, "buffa2", "a")
		//fmt.Printf("a context : %s\n", ss.PoolVocab["smalla"])
		mmm := ss.PoolVocab["smalla"].SubSpace([]int{0}).(*etensor.Float32).Values
		buff := ss.PoolVocab["buffa2"].SubSpace([]int{0}).(*etensor.Float32).Values
		for h := 0; h < (plY * plX); h++ {
			mmm[h] = buff[h]*0.5
		}
		//fmt.Printf("b type: %f\n", mmm)
		//fmt.Printf("a context : %s\n", ss.PoolVocab["smalla"])

		//pre-allocate to something
		seqid := [900]int{4, 2, 1, 3, 4, 3, 4, 3, 2, 1, 2, 4, 3, 4, 5, 6, 8, 7, 6, 5, 6, 8, 6, 5, 4, 5, 6, 5, 7, 5, 6, 5, 4, 2, 4, 2, 4, 3, 2, 3, 1, 8, 6, 8, 6, 7, 8, 1, 8, 6, 7, 6, 8, 6, 7, 5, 6, 7, 8, 1, 8, 1, 2, 3, 2, 4, 3, 2, 3, 4, 3, 1, 3, 1, 8, 7, 8, 6, 7, 8, 7, 6, 8, 7, 5, 7, 6, 8, 6, 5, 7, 6, 8, 6, 7, 8, 1, 3, 2, 4, 2, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 4, 3, 2, 4, 3, 4, 2, 1, 3, 4, 5, 4, 5, 7, 5, 7, 6, 8, 7, 5, 7, 5, 4, 2, 1, 2, 3, 4, 5, 4, 3, 4, 3, 2, 4, 2, 3, 1, 8, 7, 5, 6, 8, 6, 8, 6, 8, 1, 2, 4, 3, 2, 1, 3, 2, 4, 3, 4, 5, 6, 8, 6, 7, 6, 7, 5, 4, 3, 4, 5, 7, 8, 6, 8, 6, 7, 5, 7, 6, 5, 6, 7, 6, 8, 6, 7, 8, 6, 7, 8, 7, 5, 4, 2, 1, 3, 2, 4, 5, 6, 8, 1, 2, 1, 3, 1, 8, 6, 7, 8, 1, 8, 1, 8, 6, 7, 6, 5, 6, 5, 7, 6, 5, 4, 5, 7, 6, 5, 6, 8, 7, 5, 4, 3, 4, 5, 6, 8, 6, 7, 6, 5, 6, 5, 7, 8, 7, 8, 1, 2, 4, 5, 4, 2, 3, 4, 5, 4, 2, 4, 5, 6, 5, 4, 2, 3, 4, 3, 4, 5, 4, 2, 4, 2, 4, 3, 4, 3, 2, 4, 5, 6, 8, 7, 6, 5, 7, 8, 6, 5, 4, 2, 1, 8, 1, 3, 4, 5, 7, 6, 8, 7, 8, 7, 8, 7, 5, 7, 5, 4, 5, 4, 2, 3, 1, 3, 1, 8, 6, 5, 4, 3, 1, 2, 3, 2, 4, 3, 2, 3, 1, 2, 1, 2, 4, 5, 4, 5, 7, 5, 6, 8, 1, 3, 4, 3, 4, 5, 7, 5, 7, 5, 6, 7, 5, 7, 5, 6, 8, 7, 6, 5, 7, 6, 8, 7, 5, 4, 3, 2, 4, 5, 4, 2, 3, 4, 3, 1, 2, 4, 5, 7, 8, 6, 5, 6, 8, 7, 5, 6, 8, 7, 6, 7, 8, 7, 8, 1, 8, 1, 2, 1, 8, 1, 3, 1, 8, 1, 3, 4, 5, 7, 8, 6, 7, 6, 8, 6, 5, 6, 5, 4, 3, 2, 4, 3, 4, 3, 1, 3, 4, 2, 1, 2, 1, 3, 2, 1, 3, 4, 2, 1, 3, 1, 3, 1, 2, 3, 4, 5, 6, 5, 4, 5, 7, 8, 1, 3, 2, 4, 3, 2, 4, 5, 6, 8, 6, 8, 1, 2, 1, 2, 3, 2, 1, 3, 4, 5, 6, 7, 6, 8, 1, 2, 4, 2, 4, 3, 4, 2, 4, 2, 1, 2, 4, 2, 4, 2, 4, 2, 3, 2, 3, 4, 2, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 1, 2, 3, 2, 4, 5, 4, 5, 7, 5, 7, 8, 7, 6, 5, 6, 7, 6, 5, 6, 8, 7, 8, 6, 8, 1, 8, 6, 7, 8, 1, 3, 4, 5, 6, 5, 4, 2, 1, 8, 7, 6, 7, 5, 6, 8, 6, 7, 6, 8, 1, 2, 4, 5, 4, 2, 3, 1, 8, 6, 8, 7, 5, 4, 2, 3, 2, 1, 2, 1, 8, 7, 8, 6, 7, 5, 6, 5, 7, 8, 6, 5, 7, 6, 5, 7, 5, 4, 5, 7, 6, 8, 7, 5, 4, 3, 1, 3, 4, 5, 6, 7, 6, 7, 6, 8, 6, 7, 5, 4, 3, 4, 3, 2, 3, 2, 3, 2, 4, 2, 1, 3, 2, 1, 3, 4, 3, 2, 3, 2, 3, 2, 1, 8, 1, 8, 6, 8, 6, 7, 8, 1, 2, 1, 8, 7, 8, 1, 2, 3, 2, 4, 3, 4, 3, 2, 3, 4, 3, 1, 3, 4, 3, 2, 4, 3, 2, 1, 3, 1, 3, 2, 3, 2, 3, 2, 4, 5, 4, 5, 6, 5, 6, 7, 6, 5, 6, 5, 7, 5, 7, 6, 7, 8, 6, 8, 7, 8, 1, 2, 1, 8, 1, 2, 1, 8, 1, 2, 3, 1, 8, 6, 5, 4, 2, 1, 3, 4, 2, 3, 4, 5, 4, 2, 4, 3, 1, 3, 4, 5, 6, 7, 8, 7, 8, 6, 7, 5, 6, 8, 1, 2, 4, 5, 4, 5, 6, 8, 7, 8, 1, 2, 1, 2, 4, 2, 1, 2, 3, 4, 3, 1, 8, 6, 7, 6, 8, 6, 7, 8, 7, 5, 4, 5, 4, 5, 6, 5, 6, 5, 6, 5, 6, 7, 6, 8, 7, 8, 1, 3, 1, 2, 4, 2, 1, 8, 1, 8, 7, 5, 6, 7, 8, 7, 5, 7, 6, 5, 6, 7, 6, 7, 5, 4, 5, 4, 3, 2, 4, 5, 7, 8, 7, 6, 5, 6, 7, 8, 6, 8, 7, 5, 4, 3, 2, 3, 4, 5, 6, 8, 1, 8, 1, 3, 4, 2, 1, 8, 1, 8, 7, 5, 7, 8, 6, 8, 6, 8, 1, 3, 2, 4, 5, 4, 2, 3, 1, 3, 1, 8, 6, 8, 1, 2}

		//use random walk (2) or sequences (1). these sequences were created in MATLAB because it was way easier (see sims.m code)
		if ss.do_sequences > 1 {
			//random walk
			if ss.runnum == 0 {
				seqid = [900]int{6, 8, 7, 6, 7, 6, 7, 6, 8, 6, 8, 6, 5, 4, 2, 1, 3, 2, 1, 2, 4, 2, 4, 5, 4, 2, 4, 5, 7, 5, 4, 2, 3, 1, 2, 3, 2, 4, 3, 1, 3, 1, 3, 2, 4, 5, 7, 6, 8, 7, 6, 7, 6, 5, 7, 5, 4, 5, 7, 6, 5, 6, 8, 6, 5, 6, 7, 6, 7, 6, 8, 1, 8, 7, 5, 6, 8, 1, 3, 2, 3, 1, 8, 1, 3, 1, 3, 4, 3, 1, 8, 7, 8, 6, 8, 1, 2, 4, 3, 1, 2, 3, 1, 8, 6, 5, 6, 7, 6, 5, 4, 3, 2, 1, 3, 1, 2, 3, 2, 4, 2, 1, 3, 2, 1, 2, 1, 3, 1, 3, 1, 2, 1, 8, 1, 8, 6, 8, 7, 5, 4, 2, 3, 1, 8, 6, 7, 6, 8, 6, 7, 6, 8, 7, 6, 8, 6, 8, 1, 8, 7, 6, 8, 6, 7, 6, 8, 7, 6, 8, 7, 6, 7, 5, 7, 6, 8, 1, 8, 7, 6, 7, 5, 6, 8, 6, 5, 7, 5, 4, 5, 7, 8, 1, 8, 6, 5, 7, 5, 7, 5, 4, 5, 6, 7, 8, 6, 5, 7, 8, 1, 3, 2, 3, 2, 4, 3, 2, 3, 4, 2, 3, 1, 2, 1, 3, 4, 5, 7, 8, 1, 2, 3, 1, 8, 6, 7, 5, 4, 2, 1, 3, 1, 3, 2, 3, 4, 5, 7, 8, 7, 5, 6, 8, 7, 6, 8, 6, 5, 6, 5, 4, 2, 3, 1, 3, 1, 2, 4, 2, 1, 3, 4, 3, 2, 3, 1, 3, 1, 2, 1, 8, 6, 7, 8, 6, 7, 8, 7, 5, 6, 8, 1, 8, 1, 3, 1, 8, 7, 8, 6, 8, 6, 7, 5, 7, 8, 7, 8, 1, 8, 1, 8, 7, 5, 4, 5, 6, 5, 4, 3, 4, 3, 1, 8, 7, 6, 8, 6, 8, 7, 5, 7, 8, 6, 5, 7, 5, 6, 5, 4, 3, 1, 8, 7, 6, 5, 4, 5, 6, 7, 8, 7, 5, 4, 3, 2, 1, 3, 4, 5, 7, 6, 8, 1, 8, 7, 5, 6, 7, 6, 8, 6, 8, 1, 3, 1, 3, 1, 8, 6, 8, 6, 5, 6, 7, 8, 6, 5, 6, 8, 7, 5, 7, 8, 7, 6, 5, 7, 6, 7, 5, 6, 7, 8, 6, 8, 6, 5, 4, 3, 2, 1, 8, 1, 2, 4, 5, 6, 7, 5, 7, 6, 7, 6, 7, 6, 8, 6, 5, 7, 6, 5, 7, 8, 7, 5, 7, 8, 6, 5, 4, 3, 2, 4, 3, 1, 2, 3, 2, 1, 2, 1, 8, 1, 8, 6, 5, 7, 6, 7, 5, 4, 5, 4, 2, 3, 1, 8, 7, 8, 7, 6, 7, 8, 1, 3, 2, 1, 8, 7, 8, 1, 8, 6, 8, 1, 8, 1, 3, 2, 3, 2, 1, 3, 4, 3, 4, 3, 1, 8, 7, 6, 8, 6, 7, 8, 6, 7, 6, 5, 4, 5, 6, 8, 1, 8, 6, 5, 4, 2, 1, 8, 7, 6, 7, 8, 6, 8, 7, 8, 1, 8, 1, 8, 1, 3, 1, 8, 7, 6, 7, 5, 6, 7, 8, 7, 8, 7, 5, 6, 5, 7, 5, 7, 8, 6, 7, 6, 8, 7, 5, 7, 5, 7, 6, 5, 6, 5, 7, 8, 1, 3, 4, 3, 4, 5, 7, 5, 4, 5, 6, 7, 6, 7, 8, 1, 8, 1, 2, 1, 3, 2, 3, 2, 3, 1, 8, 7, 5, 6, 5, 4, 2, 3, 4, 5, 7, 6, 5, 4, 5, 6, 8, 1, 2, 4, 2, 1, 3, 2, 3, 2, 4, 3, 1, 8, 7, 8, 7, 5, 7, 6, 5, 7, 6, 8, 1, 8, 7, 8, 7, 6, 8, 6, 5, 4, 5, 4, 5, 7, 6, 7, 8, 6, 8, 1, 8, 7, 5, 7, 6, 8, 6, 8, 7, 5, 7, 6, 5, 6, 8, 6, 8, 7, 6, 8, 1, 3, 2, 1, 2, 4, 5, 4, 5, 4, 3, 4, 5, 7, 6, 5, 4, 2, 1, 2, 4, 5, 7, 8, 1, 2, 4, 3, 4, 2, 1, 2, 3, 4, 2, 1, 3, 2, 4, 2, 3, 2, 1, 8, 1, 8, 1, 8, 6, 5, 4, 5, 6, 5, 7, 6, 5, 7, 5, 6, 8, 1, 8, 7, 5, 6, 5, 4, 2, 4, 5, 6, 5, 6, 8, 6, 8, 1, 2, 3, 1, 3, 1, 8, 7, 8, 1, 2, 1, 8, 7, 6, 7, 6, 8, 7, 8, 7, 8, 7, 8, 7, 6, 8, 1, 8, 1, 3, 1, 3, 4, 2, 1, 3, 1, 2, 1, 8, 1, 3, 1, 8, 1, 2, 1, 8, 6, 8, 7, 8, 1, 3, 4, 2, 4, 3, 2, 1, 8, 6, 7, 8, 7, 6, 8, 7, 6, 8, 7, 5, 7, 6, 5, 7, 5, 7, 6, 7, 8, 1, 2, 4, 2, 1, 2, 1, 2, 4, 2, 3, 4, 2, 3, 2, 3, 2, 1, 8, 7, 8, 1, 2, 3, 2, 4, 3, 2, 1, 2, 1, 8, 1, 8, 6, 8, 6, 8, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 2, 4, 5, 6, 8, 7, 8, 6, 8, 1, 2, 1, 8, 7, 5, 4}
			} else if ss.runnum == 1 {
				seqid = [900]int{4, 2, 1, 3, 4, 3, 4, 3, 2, 1, 2, 4, 3, 4, 5, 6, 8, 7, 6, 5, 6, 8, 6, 5, 4, 5, 6, 5, 7, 5, 6, 5, 4, 2, 4, 2, 4, 3, 2, 3, 1, 8, 6, 8, 6, 7, 8, 1, 8, 6, 7, 6, 8, 6, 7, 5, 6, 7, 8, 1, 8, 1, 2, 3, 2, 4, 3, 2, 3, 4, 3, 1, 3, 1, 8, 7, 8, 6, 7, 8, 7, 6, 8, 7, 5, 7, 6, 8, 6, 5, 7, 6, 8, 6, 7, 8, 1, 3, 2, 4, 2, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 4, 3, 2, 4, 3, 4, 2, 1, 3, 4, 5, 4, 5, 7, 5, 7, 6, 8, 7, 5, 7, 5, 4, 2, 1, 2, 3, 4, 5, 4, 3, 4, 3, 2, 4, 2, 3, 1, 8, 7, 5, 6, 8, 6, 8, 6, 8, 1, 2, 4, 3, 2, 1, 3, 2, 4, 3, 4, 5, 6, 8, 6, 7, 6, 7, 5, 4, 3, 4, 5, 7, 8, 6, 8, 6, 7, 5, 7, 6, 5, 6, 7, 6, 8, 6, 7, 8, 6, 7, 8, 7, 5, 4, 2, 1, 3, 2, 4, 5, 6, 8, 1, 2, 1, 3, 1, 8, 6, 7, 8, 1, 8, 1, 8, 6, 7, 6, 5, 6, 5, 7, 6, 5, 4, 5, 7, 6, 5, 6, 8, 7, 5, 4, 3, 4, 5, 6, 8, 6, 7, 6, 5, 6, 5, 7, 8, 7, 8, 1, 2, 4, 5, 4, 2, 3, 4, 5, 4, 2, 4, 5, 6, 5, 4, 2, 3, 4, 3, 4, 5, 4, 2, 4, 2, 4, 3, 4, 3, 2, 4, 5, 6, 8, 7, 6, 5, 7, 8, 6, 5, 4, 2, 1, 8, 1, 3, 4, 5, 7, 6, 8, 7, 8, 7, 8, 7, 5, 7, 5, 4, 5, 4, 2, 3, 1, 3, 1, 8, 6, 5, 4, 3, 1, 2, 3, 2, 4, 3, 2, 3, 1, 2, 1, 2, 4, 5, 4, 5, 7, 5, 6, 8, 1, 3, 4, 3, 4, 5, 7, 5, 7, 5, 6, 7, 5, 7, 5, 6, 8, 7, 6, 5, 7, 6, 8, 7, 5, 4, 3, 2, 4, 5, 4, 2, 3, 4, 3, 1, 2, 4, 5, 7, 8, 6, 5, 6, 8, 7, 5, 6, 8, 7, 6, 7, 8, 7, 8, 1, 8, 1, 2, 1, 8, 1, 3, 1, 8, 1, 3, 4, 5, 7, 8, 6, 7, 6, 8, 6, 5, 6, 5, 4, 3, 2, 4, 3, 4, 3, 1, 3, 4, 2, 1, 2, 1, 3, 2, 1, 3, 4, 2, 1, 3, 1, 3, 1, 2, 3, 4, 5, 6, 5, 4, 5, 7, 8, 1, 3, 2, 4, 3, 2, 4, 5, 6, 8, 6, 8, 1, 2, 1, 2, 3, 2, 1, 3, 4, 5, 6, 7, 6, 8, 1, 2, 4, 2, 4, 3, 4, 2, 4, 2, 1, 2, 4, 2, 4, 2, 4, 2, 3, 2, 3, 4, 2, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 1, 2, 3, 2, 4, 5, 4, 5, 7, 5, 7, 8, 7, 6, 5, 6, 7, 6, 5, 6, 8, 7, 8, 6, 8, 1, 8, 6, 7, 8, 1, 3, 4, 5, 6, 5, 4, 2, 1, 8, 7, 6, 7, 5, 6, 8, 6, 7, 6, 8, 1, 2, 4, 5, 4, 2, 3, 1, 8, 6, 8, 7, 5, 4, 2, 3, 2, 1, 2, 1, 8, 7, 8, 6, 7, 5, 6, 5, 7, 8, 6, 5, 7, 6, 5, 7, 5, 4, 5, 7, 6, 8, 7, 5, 4, 3, 1, 3, 4, 5, 6, 7, 6, 7, 6, 8, 6, 7, 5, 4, 3, 4, 3, 2, 3, 2, 3, 2, 4, 2, 1, 3, 2, 1, 3, 4, 3, 2, 3, 2, 3, 2, 1, 8, 1, 8, 6, 8, 6, 7, 8, 1, 2, 1, 8, 7, 8, 1, 2, 3, 2, 4, 3, 4, 3, 2, 3, 4, 3, 1, 3, 4, 3, 2, 4, 3, 2, 1, 3, 1, 3, 2, 3, 2, 3, 2, 4, 5, 4, 5, 6, 5, 6, 7, 6, 5, 6, 5, 7, 5, 7, 6, 7, 8, 6, 8, 7, 8, 1, 2, 1, 8, 1, 2, 1, 8, 1, 2, 3, 1, 8, 6, 5, 4, 2, 1, 3, 4, 2, 3, 4, 5, 4, 2, 4, 3, 1, 3, 4, 5, 6, 7, 8, 7, 8, 6, 7, 5, 6, 8, 1, 2, 4, 5, 4, 5, 6, 8, 7, 8, 1, 2, 1, 2, 4, 2, 1, 2, 3, 4, 3, 1, 8, 6, 7, 6, 8, 6, 7, 8, 7, 5, 4, 5, 4, 5, 6, 5, 6, 5, 6, 5, 6, 7, 6, 8, 7, 8, 1, 3, 1, 2, 4, 2, 1, 8, 1, 8, 7, 5, 6, 7, 8, 7, 5, 7, 6, 5, 6, 7, 6, 7, 5, 4, 5, 4, 3, 2, 4, 5, 7, 8, 7, 6, 5, 6, 7, 8, 6, 8, 7, 5, 4, 3, 2, 3, 4, 5, 6, 8, 1, 8, 1, 3, 4, 2, 1, 8, 1, 8, 7, 5, 7, 8, 6, 8, 6, 8, 1, 3, 2, 4, 5, 4, 2, 3, 1, 3, 1, 8, 6, 8, 1, 2}
			} else if ss.runnum == 2 {
				seqid = [900]int{3, 2, 4, 3, 4, 5, 4, 3, 1, 2, 1, 2, 1, 2, 3, 2, 3, 1, 8, 1, 3, 4, 5, 6, 7, 5, 4, 3, 1, 3, 4, 2, 1, 2, 3, 4, 3, 4, 2, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 3, 1, 8, 6, 8, 1, 2, 4, 2, 4, 3, 4, 2, 3, 2, 4, 5, 4, 5, 4, 2, 4, 3, 1, 8, 6, 8, 6, 8, 7, 6, 8, 1, 3, 1, 3, 4, 3, 2, 3, 2, 1, 8, 6, 8, 6, 5, 7, 6, 7, 5, 6, 5, 4, 2, 4, 3, 2, 3, 4, 3, 2, 4, 2, 4, 2, 1, 3, 1, 2, 4, 2, 1, 8, 7, 5, 4, 2, 1, 3, 1, 2, 4, 3, 2, 1, 2, 4, 2, 4, 5, 7, 8, 7, 8, 7, 6, 7, 6, 8, 6, 7, 8, 7, 6, 5, 7, 8, 1, 2, 4, 3, 2, 4, 5, 6, 7, 6, 7, 8, 1, 2, 1, 2, 4, 3, 2, 1, 3, 1, 2, 4, 2, 4, 3, 4, 2, 3, 4, 3, 4, 2, 3, 4, 2, 3, 4, 3, 1, 3, 1, 2, 1, 8, 6, 5, 4, 2, 3, 2, 3, 2, 4, 2, 1, 8, 1, 3, 2, 3, 2, 1, 8, 7, 5, 4, 5, 7, 8, 6, 7, 6, 7, 5, 6, 5, 7, 5, 4, 5, 6, 8, 7, 8, 7, 8, 1, 2, 1, 3, 2, 4, 5, 4, 5, 6, 5, 7, 6, 8, 1, 2, 4, 3, 1, 8, 1, 3, 1, 8, 1, 8, 7, 8, 6, 8, 1, 8, 6, 5, 4, 2, 4, 3, 2, 4, 5, 4, 5, 6, 5, 6, 8, 7, 6, 8, 1, 3, 1, 2, 3, 1, 2, 1, 3, 2, 1, 8, 1, 3, 2, 4, 2, 1, 3, 2, 1, 2, 4, 5, 7, 8, 6, 7, 8, 7, 5, 6, 5, 6, 7, 6, 8, 1, 8, 1, 8, 6, 5, 6, 7, 5, 4, 3, 1, 2, 1, 3, 4, 2, 1, 8, 7, 6, 8, 7, 5, 6, 8, 1, 2, 4, 2, 4, 5, 4, 5, 4, 2, 1, 3, 1, 8, 1, 8, 6, 8, 1, 8, 7, 5, 6, 7, 8, 6, 7, 5, 6, 5, 4, 5, 4, 2, 3, 2, 1, 2, 3, 4, 2, 4, 3, 2, 1, 8, 7, 8, 6, 5, 4, 3, 1, 8, 7, 8, 1, 8, 6, 8, 7, 5, 4, 2, 3, 2, 4, 3, 1, 3, 1, 8, 6, 7, 5, 7, 5, 6, 8, 6, 5, 6, 5, 6, 7, 5, 4, 5, 6, 5, 6, 8, 1, 2, 3, 2, 3, 2, 3, 1, 3, 2, 1, 2, 4, 2, 4, 3, 2, 4, 5, 7, 5, 7, 5, 4, 2, 1, 3, 1, 3, 1, 3, 2, 4, 3, 2, 1, 3, 4, 5, 7, 5, 6, 5, 7, 8, 7, 5, 7, 6, 5, 7, 5, 4, 2, 1, 3, 4, 3, 1, 2, 4, 5, 6, 5, 4, 3, 2, 1, 8, 1, 3, 2, 1, 2, 1, 2, 1, 3, 4, 5, 6, 8, 7, 6, 5, 6, 5, 4, 5, 4, 3, 2, 1, 2, 1, 3, 1, 3, 1, 2, 1, 3, 1, 2, 4, 3, 1, 8, 6, 7, 8, 1, 8, 6, 7, 5, 7, 6, 7, 8, 7, 5, 4, 3, 2, 4, 3, 4, 3, 1, 8, 1, 3, 4, 2, 3, 4, 2, 1, 8, 1, 2, 1, 3, 2, 3, 1, 3, 4, 5, 4, 5, 7, 5, 4, 5, 6, 7, 5, 7, 6, 7, 8, 6, 8, 7, 6, 7, 6, 7, 5, 6, 5, 6, 8, 1, 8, 1, 3, 2, 3, 1, 2, 3, 1, 2, 1, 8, 6, 8, 1, 3, 2, 3, 2, 1, 8, 7, 5, 4, 2, 4, 2, 1, 2, 4, 3, 4, 5, 7, 6, 8, 7, 6, 7, 8, 7, 5, 7, 5, 4, 3, 4, 5, 4, 2, 4, 2, 1, 3, 4, 2, 4, 3, 2, 4, 3, 2, 3, 1, 3, 1, 3, 1, 8, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 8, 6, 7, 5, 7, 5, 4, 5, 4, 5, 7, 6, 5, 7, 8, 7, 6, 5, 4, 5, 6, 7, 6, 8, 6, 8, 6, 8, 1, 2, 3, 2, 1, 2, 3, 2, 3, 2, 4, 2, 3, 1, 8, 6, 8, 7, 5, 7, 8, 6, 7, 8, 1, 8, 6, 7, 8, 1, 8, 1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 5, 6, 7, 5, 6, 7, 6, 8, 7, 8, 1, 8, 1, 3, 1, 3, 2, 4, 3, 1, 2, 3, 1, 3, 4, 5, 7, 6, 5, 7, 8, 6, 8, 7, 6, 8, 7, 5, 4, 3, 2, 1, 8, 6, 7, 8, 6, 7, 6, 5, 6, 5, 4, 3, 2, 4, 5, 7, 8, 6, 7, 6, 5, 4, 3, 1, 2, 4, 5, 4, 5, 4, 3, 1, 2, 3, 1, 8, 6, 7, 5, 6, 7, 6, 7, 6, 7, 5, 4, 5, 6, 7, 8, 7, 8, 1, 3, 4, 5, 7, 5, 6, 7, 5, 6, 8, 6, 8, 6, 5, 6, 5, 7, 5, 4, 2, 3, 2, 4, 2, 1, 2, 1, 8, 7, 6}
			} else if ss.runnum == 3 {
				seqid = [900]int{1, 8, 6, 7, 6, 7, 8, 6, 7, 6, 5, 7, 5, 7, 5, 6, 5, 6, 8, 1, 2, 3, 1, 8, 7, 8, 6, 5, 7, 6, 7, 8, 6, 5, 7, 8, 6, 7, 8, 1, 3, 1, 3, 1, 8, 1, 8, 7, 5, 4, 2, 1, 3, 2, 4, 2, 1, 3, 1, 2, 3, 1, 3, 2, 3, 1, 3, 1, 3, 4, 5, 6, 7, 5, 6, 7, 6, 8, 7, 8, 7, 6, 8, 1, 2, 3, 2, 1, 2, 3, 4, 2, 4, 5, 6, 5, 7, 6, 5, 7, 6, 7, 6, 5, 4, 3, 1, 8, 7, 6, 8, 6, 8, 6, 8, 7, 8, 6, 5, 6, 8, 1, 2, 1, 2, 1, 8, 6, 8, 1, 3, 1, 2, 4, 2, 4, 5, 6, 5, 7, 6, 8, 1, 3, 4, 2, 4, 3, 4, 5, 7, 8, 1, 3, 1, 3, 2, 4, 2, 3, 4, 5, 6, 8, 1, 3, 2, 3, 2, 3, 2, 3, 4, 5, 6, 7, 6, 7, 8, 1, 2, 4, 5, 6, 5, 7, 5, 4, 3, 4, 3, 4, 5, 6, 8, 7, 5, 4, 3, 4, 3, 2, 3, 2, 1, 2, 3, 2, 3, 2, 3, 4, 5, 6, 8, 6, 8, 7, 6, 7, 5, 7, 6, 5, 6, 5, 4, 5, 7, 8, 7, 5, 6, 5, 7, 6, 5, 7, 5, 6, 8, 1, 8, 1, 2, 3, 2, 1, 3, 1, 3, 4, 3, 2, 4, 3, 2, 3, 1, 8, 6, 8, 1, 8, 1, 2, 4, 2, 1, 3, 4, 2, 4, 2, 1, 8, 7, 8, 1, 8, 7, 5, 4, 2, 1, 2, 3, 2, 1, 2, 4, 5, 4, 2, 1, 3, 4, 5, 6, 5, 4, 3, 4, 2, 3, 1, 3, 4, 3, 1, 3, 2, 1, 3, 1, 8, 6, 7, 5, 7, 5, 4, 2, 3, 2, 4, 3, 1, 8, 1, 8, 7, 6, 7, 8, 1, 8, 6, 5, 6, 7, 5, 4, 5, 4, 5, 4, 2, 1, 2, 1, 3, 4, 3, 1, 3, 2, 1, 8, 6, 7, 8, 6, 5, 7, 5, 6, 8, 7, 8, 7, 5, 7, 5, 7, 5, 7, 8, 1, 8, 7, 5, 7, 8, 1, 2, 1, 3, 4, 3, 1, 2, 3, 2, 3, 4, 5, 7, 6, 5, 4, 3, 2, 3, 4, 5, 7, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 5, 4, 3, 4, 3, 4, 5, 4, 5, 7, 8, 7, 5, 4, 2, 1, 8, 1, 2, 4, 3, 2, 1, 3, 1, 2, 3, 2, 4, 2, 1, 3, 4, 3, 1, 3, 1, 3, 2, 3, 4, 2, 1, 3, 2, 1, 2, 4, 3, 4, 2, 4, 2, 3, 2, 3, 4, 5, 7, 8, 1, 2, 4, 3, 4, 2, 3, 4, 3, 1, 3, 1, 2, 4, 3, 2, 1, 8, 7, 6, 7, 8, 6, 8, 6, 5, 7, 6, 7, 5, 4, 3, 1, 2, 1, 3, 1, 8, 7, 5, 6, 8, 7, 6, 8, 6, 5, 6, 5, 7, 6, 8, 6, 7, 5, 6, 7, 8, 7, 5, 4, 2, 3, 1, 3, 2, 4, 5, 4, 3, 1, 3, 4, 2, 4, 5, 7, 6, 8, 6, 7, 5, 6, 5, 6, 5, 7, 8, 6, 5, 6, 7, 8, 7, 8, 6, 7, 6, 7, 6, 7, 5, 7, 8, 6, 5, 7, 5, 6, 7, 8, 7, 8, 1, 2, 1, 8, 1, 8, 1, 2, 3, 1, 3, 1, 8, 7, 8, 6, 7, 8, 6, 7, 5, 4, 3, 1, 3, 1, 3, 4, 2, 3, 1, 3, 1, 8, 7, 6, 5, 4, 5, 6, 5, 7, 6, 8, 1, 2, 1, 8, 6, 5, 6, 7, 6, 8, 6, 8, 7, 5, 6, 7, 6, 5, 7, 6, 5, 4, 2, 4, 3, 4, 2, 3, 2, 4, 5, 6, 7, 5, 4, 2, 4, 3, 1, 3, 1, 2, 1, 8, 7, 8, 7, 8, 6, 5, 7, 8, 6, 8, 6, 7, 5, 6, 7, 6, 7, 6, 8, 6, 7, 5, 7, 5, 6, 8, 7, 5, 7, 5, 6, 8, 1, 8, 1, 8, 7, 6, 7, 5, 4, 2, 4, 5, 6, 7, 8, 1, 2, 1, 8, 7, 8, 6, 8, 1, 3, 4, 5, 7, 5, 7, 5, 4, 5, 7, 5, 7, 5, 6, 5, 4, 2, 4, 5, 6, 7, 6, 7, 8, 6, 8, 1, 3, 2, 4, 3, 2, 1, 2, 1, 8, 1, 2, 3, 2, 3, 2, 4, 2, 1, 8, 1, 2, 3, 1, 8, 1, 3, 1, 8, 6, 8, 7, 5, 7, 5, 7, 8, 1, 8, 7, 5, 4, 5, 7, 5, 7, 8, 1, 2, 3, 4, 3, 2, 4, 3, 4, 2, 4, 2, 1, 8, 1, 3, 2, 1, 3, 1, 3, 1, 3, 2, 4, 2, 3, 4, 3, 2, 3, 4, 3, 1, 3, 2, 4, 3, 2, 4, 5, 4, 3, 1, 3, 2, 3, 1, 8, 7, 6, 5, 6, 5, 4, 5, 6, 5, 7, 8, 7, 8, 7, 8, 7, 5, 6, 5, 4, 5, 6, 7, 6, 8, 7, 8, 7, 5, 7, 6, 8, 1, 3, 1, 8, 6, 7, 8, 7, 5, 7, 8, 7}
			} else if ss.runnum == 4 {
				seqid = [900]int{1, 2, 3, 2, 4, 5, 6, 5, 7, 5, 7, 8, 6, 8, 7, 5, 4, 3, 1, 8, 6, 7, 6, 5, 4, 3, 4, 3, 4, 5, 6, 5, 7, 5, 7, 8, 1, 8, 6, 5, 4, 5, 4, 5, 6, 8, 6, 7, 6, 7, 6, 7, 6, 8, 7, 6, 8, 7, 6, 8, 1, 2, 3, 2, 4, 3, 1, 8, 7, 6, 8, 6, 8, 6, 5, 4, 5, 6, 5, 4, 2, 1, 8, 1, 3, 4, 2, 1, 3, 4, 5, 6, 7, 5, 4, 2, 1, 8, 7, 5, 7, 6, 8, 1, 8, 1, 8, 6, 5, 7, 8, 6, 7, 8, 6, 5, 4, 2, 1, 2, 1, 2, 4, 2, 4, 5, 6, 8, 7, 8, 6, 7, 6, 5, 6, 5, 7, 6, 7, 8, 7, 6, 5, 4, 5, 6, 8, 7, 8, 1, 3, 2, 1, 3, 4, 3, 2, 1, 8, 7, 5, 7, 6, 5, 6, 8, 1, 8, 1, 2, 1, 8, 6, 7, 8, 7, 8, 1, 3, 4, 2, 3, 1, 8, 7, 5, 7, 6, 7, 6, 7, 5, 7, 5, 7, 8, 1, 2, 4, 2, 1, 3, 2, 1, 3, 2, 4, 3, 1, 2, 3, 4, 3, 1, 3, 2, 4, 3, 2, 1, 2, 4, 2, 1, 8, 1, 2, 4, 5, 7, 8, 1, 3, 4, 3, 4, 3, 1, 2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 3, 2, 3, 2, 1, 8, 7, 5, 6, 7, 5, 7, 8, 7, 5, 7, 6, 7, 6, 8, 7, 8, 1, 8, 6, 5, 7, 5, 4, 5, 4, 2, 4, 2, 4, 2, 3, 1, 3, 2, 1, 8, 7, 6, 7, 5, 4, 5, 7, 6, 5, 7, 8, 1, 8, 6, 5, 7, 6, 8, 7, 5, 4, 2, 3, 1, 2, 1, 8, 7, 8, 1, 3, 1, 2, 1, 8, 1, 8, 7, 8, 7, 6, 7, 6, 7, 6, 5, 7, 5, 6, 5, 7, 8, 1, 8, 1, 2, 1, 2, 4, 5, 7, 8, 6, 5, 6, 7, 6, 7, 5, 6, 8, 6, 8, 1, 3, 4, 5, 7, 8, 1, 8, 7, 5, 4, 5, 4, 3, 2, 3, 1, 3, 4, 5, 6, 8, 6, 7, 8, 7, 5, 6, 7, 6, 8, 1, 8, 6, 5, 6, 7, 8, 6, 8, 1, 8, 6, 5, 6, 5, 7, 8, 1, 8, 1, 2, 4, 2, 1, 8, 7, 5, 7, 6, 5, 7, 5, 6, 7, 6, 8, 1, 8, 1, 2, 1, 2, 4, 5, 7, 8, 6, 7, 8, 1, 8, 6, 8, 1, 3, 1, 2, 3, 2, 4, 3, 2, 4, 5, 7, 8, 7, 8, 1, 2, 3, 4, 2, 3, 2, 3, 2, 3, 4, 5, 4, 3, 1, 2, 4, 5, 7, 6, 5, 4, 3, 1, 3, 2, 4, 5, 4, 5, 6, 7, 8, 6, 8, 6, 8, 7, 5, 4, 3, 2, 3, 2, 3, 2, 4, 3, 1, 3, 2, 3, 2, 1, 3, 4, 2, 1, 3, 1, 8, 1, 8, 1, 8, 6, 7, 5, 4, 2, 4, 5, 4, 5, 4, 5, 4, 5, 6, 5, 4, 2, 1, 3, 4, 2, 1, 3, 1, 8, 6, 5, 6, 5, 6, 7, 5, 7, 6, 5, 4, 5, 7, 6, 5, 6, 8, 1, 8, 1, 2, 1, 8, 6, 5, 6, 5, 4, 3, 1, 8, 6, 8, 7, 5, 6, 5, 7, 8, 7, 5, 4, 3, 4, 2, 1, 3, 4, 3, 2, 4, 3, 4, 3, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 5, 6, 7, 8, 6, 7, 8, 1, 8, 6, 8, 6, 5, 6, 5, 7, 5, 4, 3, 4, 5, 4, 2, 3, 4, 3, 1, 2, 1, 3, 1, 2, 4, 5, 4, 2, 4, 3, 4, 2, 4, 5, 6, 5, 7, 5, 7, 8, 6, 7, 6, 8, 6, 8, 6, 5, 7, 5, 6, 8, 7, 5, 7, 6, 8, 7, 6, 7, 6, 5, 6, 5, 7, 8, 6, 7, 6, 7, 6, 8, 7, 8, 7, 5, 4, 5, 4, 2, 3, 4, 3, 1, 3, 2, 1, 8, 6, 8, 7, 5, 4, 2, 4, 3, 4, 2, 3, 1, 3, 4, 2, 3, 4, 3, 2, 3, 1, 3, 1, 8, 6, 5, 7, 5, 4, 2, 1, 2, 1, 8, 1, 8, 1, 8, 7, 5, 6, 5, 4, 2, 3, 2, 3, 2, 4, 3, 2, 3, 2, 4, 2, 3, 2, 1, 3, 4, 3, 4, 5, 7, 5, 6, 5, 6, 7, 5, 7, 5, 6, 7, 8, 1, 2, 1, 3, 4, 5, 4, 2, 3, 4, 5, 4, 2, 4, 2, 4, 3, 1, 2, 1, 3, 2, 3, 4, 2, 3, 4, 2, 3, 1, 3, 4, 2, 4, 5, 6, 7, 6, 8, 6, 8, 7, 5, 6, 8, 1, 2, 3, 1, 8, 6, 8, 1, 2, 3, 1, 2, 3, 2, 4, 5, 7, 8, 1, 8, 1, 8, 6, 5, 7, 5, 6, 5, 4, 2, 3, 1, 8, 1, 8, 6, 8, 7, 8, 1, 3, 1, 2, 4, 3, 4, 2, 1, 8, 1, 3, 2, 4, 3, 4, 3, 4, 3, 2, 4, 3, 2, 4, 5, 6, 5, 4, 3}
			} else if ss.runnum == 5 {
				seqid = [900]int{2, 3, 2, 3, 1, 3, 2, 1, 8, 7, 6, 7, 5, 7, 8, 1, 2, 4, 5, 4, 2, 1, 3, 1, 8, 6, 8, 7, 6, 7, 6, 8, 6, 8, 7, 6, 7, 5, 6, 7, 6, 8, 1, 8, 7, 5, 6, 8, 7, 8, 7, 8, 7, 5, 7, 5, 4, 5, 4, 5, 7, 5, 7, 8, 6, 5, 4, 3, 2, 3, 4, 3, 2, 4, 2, 3, 1, 8, 7, 6, 7, 5, 6, 5, 7, 5, 4, 5, 6, 5, 6, 7, 8, 1, 8, 6, 8, 6, 7, 8, 1, 8, 6, 8, 6, 7, 6, 8, 7, 6, 8, 6, 7, 6, 5, 7, 6, 8, 7, 8, 1, 3, 2, 1, 2, 4, 5, 4, 2, 3, 4, 2, 4, 3, 4, 5, 4, 2, 3, 2, 1, 3, 2, 3, 2, 3, 4, 3, 2, 4, 2, 1, 3, 4, 3, 1, 2, 4, 2, 3, 4, 2, 3, 1, 2, 1, 3, 2, 4, 2, 4, 3, 1, 2, 1, 3, 4, 2, 4, 2, 3, 1, 3, 4, 3, 2, 1, 8, 6, 8, 1, 3, 4, 5, 6, 8, 1, 3, 1, 3, 1, 2, 3, 1, 8, 1, 8, 6, 5, 7, 8, 6, 7, 8, 1, 2, 1, 2, 4, 5, 7, 6, 8, 7, 5, 7, 8, 6, 5, 4, 5, 7, 6, 8, 6, 7, 5, 4, 5, 7, 8, 7, 8, 1, 2, 3, 1, 2, 3, 1, 2, 4, 3, 1, 3, 2, 3, 1, 8, 1, 2, 3, 4, 5, 4, 2, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 1, 2, 4, 5, 6, 8, 1, 2, 4, 3, 2, 1, 2, 1, 3, 4, 2, 4, 5, 7, 6, 7, 5, 6, 8, 1, 8, 6, 8, 7, 5, 7, 6, 7, 8, 6, 7, 5, 6, 7, 8, 1, 3, 1, 2, 4, 3, 4, 5, 6, 8, 1, 3, 1, 3, 2, 3, 4, 3, 2, 1, 8, 6, 5, 6, 8, 6, 5, 6, 5, 4, 5, 4, 5, 4, 5, 4, 3, 4, 2, 1, 8, 6, 8, 7, 8, 1, 3, 1, 3, 4, 2, 3, 2, 1, 2, 4, 2, 4, 3, 2, 3, 2, 1, 8, 7, 6, 7, 5, 6, 8, 6, 5, 6, 8, 7, 6, 8, 7, 5, 6, 5, 6, 7, 5, 4, 5, 4, 2, 4, 5, 6, 5, 4, 2, 1, 2, 4, 3, 4, 2, 4, 5, 7, 6, 5, 7, 5, 6, 8, 1, 2, 3, 1, 3, 1, 3, 1, 8, 7, 6, 5, 7, 8, 7, 6, 8, 6, 7, 5, 7, 8, 7, 8, 7, 5, 6, 7, 8, 7, 5, 4, 5, 4, 5, 4, 3, 2, 4, 2, 3, 1, 8, 7, 6, 8, 6, 7, 8, 6, 8, 7, 8, 6, 8, 6, 5, 7, 5, 4, 2, 4, 5, 7, 6, 5, 7, 8, 6, 8, 7, 5, 7, 5, 6, 8, 1, 8, 7, 6, 5, 6, 7, 5, 4, 5, 7, 8, 7, 8, 1, 2, 3, 4, 3, 1, 3, 4, 5, 6, 5, 4, 2, 4, 3, 2, 1, 2, 4, 2, 3, 1, 3, 1, 2, 1, 3, 1, 8, 6, 7, 5, 4, 2, 3, 4, 2, 4, 2, 3, 2, 4, 2, 1, 2, 3, 4, 2, 4, 2, 3, 4, 3, 2, 3, 2, 1, 2, 4, 5, 7, 5, 6, 8, 6, 5, 4, 3, 1, 8, 6, 5, 7, 6, 5, 6, 8, 7, 8, 1, 8, 1, 8, 1, 2, 1, 8, 6, 8, 1, 8, 1, 3, 1, 3, 1, 8, 1, 8, 6, 8, 1, 3, 4, 2, 4, 2, 3, 4, 2, 4, 3, 4, 2, 4, 3, 4, 5, 6, 5, 4, 2, 4, 2, 4, 5, 7, 8, 7, 6, 8, 6, 8, 7, 8, 7, 5, 4, 5, 6, 5, 4, 3, 4, 2, 4, 5, 6, 5, 4, 5, 7, 5, 6, 7, 6, 8, 6, 7, 5, 6, 5, 4, 3, 2, 3, 1, 2, 4, 2, 4, 2, 4, 3, 1, 2, 1, 2, 1, 8, 6, 7, 6, 8, 1, 3, 1, 2, 1, 2, 3, 4, 3, 4, 5, 7, 5, 6, 5, 6, 8, 1, 2, 1, 3, 4, 5, 6, 5, 7, 5, 6, 5, 7, 5, 7, 5, 4, 2, 4, 2, 4, 2, 3, 1, 3, 1, 3, 4, 3, 2, 4, 3, 2, 1, 8, 1, 3, 4, 2, 4, 5, 6, 8, 1, 3, 4, 5, 4, 5, 6, 8, 6, 7, 5, 7, 8, 6, 7, 6, 8, 1, 2, 3, 2, 1, 8, 7, 6, 8, 7, 5, 6, 5, 6, 5, 6, 8, 6, 7, 5, 6, 7, 8, 7, 6, 5, 6, 8, 7, 6, 7, 5, 6, 8, 7, 6, 5, 6, 7, 8, 1, 3, 4, 3, 4, 3, 4, 2, 1, 8, 7, 8, 7, 5, 7, 5, 6, 5, 7, 8, 7, 5, 7, 6, 7, 6, 8, 7, 8, 7, 5, 6, 5, 4, 5, 7, 8, 7, 5, 4, 5, 4, 3, 4, 2, 3, 1, 8, 6, 5, 6, 8, 6, 7, 8, 7, 6, 7, 8, 6, 7, 5, 7, 8, 7, 8, 1, 3, 1, 2, 1, 8, 7, 5, 4, 3, 2, 3, 4, 5, 6, 8}
			} else if ss.runnum == 6 {
				seqid = [900]int{2, 3, 2, 3, 1, 3, 1, 8, 1, 2, 1, 3, 1, 2, 3, 4, 5, 7, 6, 8, 6, 8, 6, 5, 6, 5, 6, 8, 1, 3, 2, 1, 3, 2, 1, 2, 1, 3, 1, 3, 1, 3, 1, 2, 3, 4, 2, 4, 2, 3, 4, 5, 4, 3, 4, 2, 4, 2, 3, 4, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3, 2, 4, 5, 6, 8, 7, 5, 6, 8, 1, 3, 1, 2, 1, 2, 3, 1, 2, 4, 2, 3, 1, 3, 4, 3, 1, 3, 2, 3, 2, 3, 2, 3, 1, 8, 7, 5, 7, 6, 8, 6, 7, 8, 1, 3, 1, 8, 1, 3, 1, 8, 6, 8, 6, 7, 6, 7, 6, 5, 7, 5, 4, 2, 1, 3, 1, 3, 4, 2, 4, 3, 1, 2, 1, 3, 2, 1, 3, 1, 2, 4, 3, 2, 4, 2, 4, 3, 1, 3, 2, 3, 4, 5, 6, 7, 8, 1, 8, 6, 7, 8, 7, 6, 5, 6, 7, 8, 7, 6, 5, 6, 8, 6, 5, 4, 5, 7, 5, 7, 8, 6, 8, 7, 6, 7, 8, 1, 8, 6, 7, 5, 6, 7, 8, 6, 8, 6, 7, 8, 7, 5, 7, 6, 8, 7, 6, 7, 6, 5, 4, 2, 3, 2, 3, 2, 1, 2, 3, 2, 4, 3, 4, 5, 4, 3, 2, 3, 4, 5, 7, 5, 6, 5, 4, 2, 3, 4, 3, 4, 5, 6, 5, 4, 5, 7, 6, 8, 6, 5, 4, 2, 1, 2, 3, 2, 1, 3, 2, 4, 2, 4, 5, 4, 3, 4, 5, 7, 5, 4, 5, 6, 8, 7, 5, 4, 2, 3, 1, 2, 1, 2, 3, 1, 2, 4, 5, 7, 6, 5, 7, 8, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 4, 2, 1, 8, 6, 5, 6, 5, 6, 7, 8, 7, 8, 6, 5, 7, 6, 8, 6, 8, 7, 8, 6, 5, 7, 5, 4, 2, 4, 3, 1, 2, 1, 3, 2, 1, 8, 7, 6, 8, 6, 8, 1, 3, 2, 3, 1, 8, 6, 8, 6, 8, 6, 7, 5, 4, 2, 1, 2, 3, 4, 5, 6, 8, 1, 8, 1, 8, 7, 5, 7, 6, 8, 1, 2, 1, 2, 3, 4, 5, 6, 5, 7, 5, 7, 6, 5, 4, 3, 4, 2, 4, 3, 1, 2, 1, 8, 7, 6, 5, 7, 8, 6, 5, 6, 7, 5, 6, 7, 5, 7, 6, 8, 6, 8, 1, 8, 1, 3, 1, 3, 1, 2, 3, 4, 5, 4, 5, 4, 2, 3, 2, 3, 1, 8, 1, 8, 1, 8, 1, 8, 7, 6, 5, 4, 3, 1, 8, 6, 5, 6, 8, 6, 8, 6, 8, 7, 8, 6, 7, 8, 6, 7, 8, 1, 8, 1, 2, 1, 3, 1, 3, 1, 2, 1, 3, 2, 3, 1, 2, 4, 3, 2, 3, 4, 5, 6, 5, 4, 3, 1, 8, 1, 2, 4, 5, 7, 8, 7, 8, 7, 6, 8, 6, 7, 6, 8, 7, 5, 6, 8, 7, 5, 4, 2, 4, 3, 2, 4, 2, 4, 5, 4, 2, 4, 5, 6, 5, 6, 8, 7, 6, 5, 6, 5, 7, 5, 4, 3, 2, 3, 1, 8, 1, 2, 1, 2, 3, 4, 2, 4, 2, 4, 2, 3, 4, 3, 2, 3, 4, 3, 4, 2, 1, 8, 1, 2, 1, 3, 4, 3, 4, 3, 1, 3, 4, 5, 7, 8, 1, 3, 4, 2, 1, 3, 4, 2, 4, 3, 1, 8, 1, 3, 4, 3, 2, 1, 3, 1, 3, 2, 4, 5, 7, 6, 8, 7, 5, 6, 7, 8, 1, 8, 7, 8, 6, 5, 4, 3, 2, 4, 3, 2, 3, 4, 2, 3, 2, 4, 2, 4, 3, 1, 2, 3, 4, 5, 6, 7, 5, 7, 8, 1, 2, 4, 3, 1, 3, 4, 3, 1, 2, 3, 4, 3, 2, 1, 8, 1, 2, 4, 2, 1, 3, 2, 3, 1, 2, 1, 8, 1, 8, 6, 8, 7, 6, 8, 7, 8, 1, 3, 2, 4, 5, 4, 2, 4, 5, 6, 8, 1, 8, 6, 8, 7, 8, 6, 8, 7, 6, 7, 8, 1, 3, 4, 2, 1, 3, 1, 8, 7, 5, 6, 5, 7, 5, 6, 8, 6, 5, 6, 7, 5, 4, 3, 1, 8, 7, 5, 6, 7, 5, 7, 8, 6, 8, 6, 8, 1, 2, 1, 8, 7, 5, 6, 7, 8, 7, 5, 6, 7, 8, 7, 6, 8, 7, 8, 6, 5, 7, 6, 8, 6, 7, 8, 7, 8, 7, 8, 6, 5, 7, 6, 5, 6, 7, 6, 8, 6, 7, 5, 7, 5, 6, 8, 7, 8, 7, 6, 8, 1, 2, 1, 3, 4, 2, 1, 8, 7, 6, 7, 8, 6, 8, 1, 3, 4, 2, 4, 5, 7, 5, 7, 6, 8, 6, 5, 6, 5, 4, 2, 4, 5, 4, 2, 4, 2, 1, 2, 1, 8, 7, 5, 6, 8, 1, 3, 2, 1, 8, 6, 5, 4, 3, 1, 8, 1, 8, 6, 8, 7, 6, 8, 7, 8, 6, 8, 7, 5, 4, 2, 1, 8, 1, 2, 1, 3, 4, 3, 1, 8, 7, 5, 6, 7, 6, 8, 6, 7, 8, 6, 8, 7, 8, 6, 8, 7, 5, 6}
			} else if ss.runnum == 7 {
				seqid = [900]int{1, 2, 4, 2, 4, 2, 1, 2, 1, 3, 4, 3, 4, 3, 1, 3, 4, 2, 3, 1, 8, 7, 5, 4, 3, 2, 1, 2, 4, 5, 4, 2, 4, 5, 4, 3, 2, 1, 2, 1, 8, 1, 2, 3, 1, 8, 7, 8, 1, 3, 4, 3, 1, 2, 4, 3, 4, 5, 6, 8, 6, 8, 6, 5, 7, 8, 6, 7, 8, 6, 7, 6, 8, 6, 7, 5, 6, 8, 1, 3, 4, 2, 1, 3, 2, 1, 3, 1, 2, 1, 2, 1, 8, 1, 2, 4, 5, 7, 6, 7, 6, 8, 1, 8, 6, 8, 1, 3, 2, 1, 3, 2, 3, 1, 2, 4, 2, 3, 2, 1, 3, 2, 1, 2, 3, 2, 3, 1, 3, 1, 8, 1, 8, 7, 8, 1, 3, 2, 4, 3, 4, 3, 1, 3, 2, 3, 2, 1, 2, 3, 4, 5, 7, 8, 1, 2, 3, 2, 1, 2, 1, 3, 4, 3, 2, 1, 2, 1, 3, 1, 2, 1, 8, 6, 8, 6, 5, 4, 5, 7, 5, 7, 8, 7, 6, 5, 6, 7, 8, 1, 2, 1, 3, 1, 2, 4, 5, 7, 5, 6, 7, 5, 7, 5, 7, 6, 7, 6, 8, 6, 8, 1, 8, 6, 5, 7, 8, 7, 8, 1, 2, 4, 5, 6, 5, 4, 2, 3, 1, 3, 1, 3, 4, 3, 1, 3, 4, 5, 6, 8, 6, 7, 8, 7, 8, 7, 8, 7, 8, 1, 2, 3, 2, 3, 2, 1, 2, 1, 8, 7, 6, 7, 6, 8, 6, 8, 6, 7, 8, 7, 5, 7, 5, 4, 3, 4, 5, 4, 5, 6, 7, 5, 4, 5, 6, 8, 1, 8, 7, 8, 1, 8, 6, 5, 4, 2, 3, 4, 3, 4, 2, 4, 3, 4, 2, 4, 3, 1, 2, 4, 5, 7, 8, 7, 6, 8, 1, 2, 3, 4, 5, 4, 2, 4, 5, 6, 8, 7, 5, 7, 8, 7, 6, 8, 1, 8, 7, 5, 6, 7, 8, 7, 6, 8, 7, 6, 8, 1, 8, 7, 6, 8, 6, 8, 7, 8, 7, 6, 7, 5, 4, 2, 1, 2, 1, 3, 4, 5, 6, 8, 7, 8, 1, 2, 3, 4, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6, 8, 6, 5, 4, 5, 6, 8, 7, 8, 7, 5, 6, 5, 6, 5, 4, 2, 3, 1, 8, 7, 5, 4, 2, 1, 3, 2, 3, 2, 4, 3, 1, 8, 7, 5, 6, 8, 7, 8, 7, 8, 6, 7, 6, 8, 6, 7, 8, 6, 5, 7, 8, 7, 6, 8, 6, 8, 1, 8, 1, 2, 3, 2, 4, 3, 1, 3, 2, 1, 8, 6, 8, 7, 5, 7, 8, 6, 7, 6, 5, 7, 6, 5, 4, 3, 2, 3, 1, 8, 7, 8, 6, 5, 4, 2, 3, 1, 2, 1, 3, 1, 8, 6, 5, 6, 8, 6, 8, 1, 2, 3, 2, 3, 4, 2, 4, 5, 6, 8, 6, 5, 6, 8, 1, 8, 1, 8, 6, 7, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 6, 8, 7, 5, 4, 5, 6, 7, 5, 6, 8, 7, 8, 7, 6, 7, 6, 7, 6, 8, 1, 2, 4, 5, 4, 5, 6, 8, 7, 8, 7, 5, 6, 8, 6, 7, 6, 8, 7, 5, 6, 7, 5, 7, 6, 5, 7, 5, 4, 5, 4, 3, 2, 3, 4, 3, 1, 8, 6, 8, 7, 5, 6, 8, 1, 8, 6, 7, 5, 6, 5, 6, 8, 1, 2, 3, 4, 2, 4, 5, 4, 3, 4, 5, 6, 5, 4, 5, 7, 5, 7, 5, 6, 8, 1, 3, 2, 4, 5, 6, 5, 7, 5, 4, 3, 4, 3, 1, 2, 4, 2, 3, 1, 3, 1, 8, 6, 8, 6, 7, 8, 6, 5, 7, 5, 7, 6, 5, 4, 2, 4, 3, 1, 8, 1, 3, 4, 2, 4, 2, 1, 2, 4, 5, 7, 6, 7, 8, 1, 2, 3, 2, 4, 2, 3, 2, 3, 1, 8, 1, 2, 3, 4, 5, 4, 2, 4, 5, 4, 2, 3, 1, 2, 3, 4, 5, 6, 8, 6, 5, 6, 5, 7, 5, 6, 7, 6, 8, 6, 8, 7, 8, 1, 8, 6, 5, 4, 2, 3, 1, 8, 6, 8, 1, 2, 3, 2, 3, 2, 4, 2, 4, 3, 1, 2, 4, 5, 7, 8, 7, 8, 7, 6, 7, 8, 6, 5, 6, 8, 1, 3, 4, 2, 1, 2, 1, 3, 4, 3, 4, 2, 4, 5, 4, 5, 4, 3, 4, 3, 2, 1, 3, 4, 2, 1, 2, 1, 8, 1, 8, 6, 5, 7, 5, 7, 8, 6, 7, 5, 4, 5, 7, 8, 6, 7, 8, 1, 2, 3, 1, 8, 6, 7, 5, 6, 8, 6, 7, 5, 4, 3, 1, 8, 1, 2, 4, 3, 1, 3, 2, 1, 2, 4, 3, 2, 4, 5, 7, 8, 6, 8, 7, 5, 7, 8, 6, 7, 8, 7, 8, 1, 3, 1, 2, 3, 2, 3, 4, 5, 7, 8, 6, 7, 5, 7, 6, 5, 6, 5, 4, 2, 3, 2, 4, 5, 7, 8, 6, 5, 4, 2, 1, 2, 4, 3, 2, 3, 2, 4, 2, 4, 5, 6, 7, 8, 6, 8, 1, 2, 3, 2, 1, 3, 1, 2}
			} else if ss.runnum == 8 {
				seqid = [900]int{1, 8, 7, 5, 4, 5, 4, 5, 6, 5, 7, 6, 8, 7, 6, 5, 4, 5, 4, 3, 4, 2, 4, 2, 1, 2, 4, 2, 3, 1, 3, 1, 3, 4, 2, 3, 1, 3, 2, 1, 3, 1, 8, 7, 8, 1, 3, 2, 1, 8, 1, 8, 1, 2, 4, 2, 3, 2, 1, 3, 1, 3, 4, 3, 1, 3, 4, 5, 4, 2, 1, 8, 7, 5, 6, 7, 6, 5, 7, 8, 1, 8, 1, 8, 1, 2, 1, 8, 1, 3, 4, 2, 4, 5, 6, 8, 6, 5, 7, 6, 7, 8, 7, 8, 7, 5, 6, 8, 7, 5, 6, 7, 5, 6, 5, 6, 5, 7, 6, 8, 6, 7, 8, 6, 7, 5, 4, 2, 4, 5, 7, 8, 7, 6, 5, 4, 5, 7, 6, 5, 6, 8, 7, 5, 4, 3, 1, 2, 3, 4, 5, 7, 6, 7, 8, 1, 8, 7, 8, 7, 5, 6, 5, 7, 5, 7, 8, 6, 7, 6, 7, 5, 7, 8, 1, 8, 7, 6, 5, 6, 5, 4, 2, 3, 4, 3, 4, 5, 7, 5, 6, 8, 7, 5, 6, 5, 6, 7, 5, 7, 8, 7, 8, 1, 8, 1, 2, 3, 4, 3, 1, 3, 4, 2, 4, 2, 3, 2, 4, 2, 3, 4, 2, 4, 3, 2, 1, 3, 1, 3, 1, 2, 4, 2, 4, 5, 7, 5, 6, 8, 7, 5, 4, 5, 6, 7, 8, 1, 3, 1, 2, 1, 3, 1, 3, 2, 4, 2, 1, 2, 1, 3, 1, 2, 3, 4, 5, 7, 8, 1, 8, 7, 6, 8, 6, 7, 6, 8, 6, 7, 6, 8, 7, 5, 4, 2, 1, 2, 3, 4, 2, 4, 3, 2, 3, 2, 4, 2, 4, 2, 1, 3, 4, 2, 3, 1, 3, 4, 3, 4, 3, 2, 3, 2, 1, 2, 4, 5, 4, 5, 4, 2, 3, 2, 3, 4, 5, 6, 8, 6, 8, 6, 7, 8, 6, 8, 1, 3, 2, 4, 5, 7, 8, 7, 8, 1, 3, 2, 1, 2, 4, 3, 4, 3, 4, 2, 3, 2, 1, 2, 4, 2, 4, 2, 4, 2, 4, 3, 1, 2, 1, 2, 1, 8, 1, 3, 4, 5, 7, 6, 8, 6, 8, 6, 7, 6, 5, 7, 8, 6, 5, 7, 6, 5, 6, 8, 7, 5, 6, 5, 6, 5, 6, 7, 8, 7, 5, 6, 7, 8, 7, 8, 7, 6, 5, 4, 5, 4, 3, 4, 2, 4, 3, 2, 3, 2, 3, 4, 3, 1, 2, 4, 3, 2, 1, 3, 1, 2, 3, 2, 1, 8, 7, 5, 7, 5, 7, 6, 8, 1, 2, 4, 5, 7, 6, 8, 6, 5, 6, 8, 7, 8, 6, 7, 8, 6, 7, 6, 7, 5, 4, 5, 7, 8, 1, 8, 6, 7, 6, 7, 6, 8, 7, 6, 5, 7, 6, 7, 5, 6, 7, 5, 4, 5, 4, 5, 7, 8, 6, 7, 6, 8, 7, 6, 7, 8, 1, 8, 7, 5, 7, 5, 4, 3, 1, 8, 1, 3, 2, 1, 8, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 3, 4, 5, 7, 8, 7, 6, 5, 7, 8, 1, 2, 1, 8, 1, 3, 4, 2, 1, 3, 2, 3, 1, 2, 1, 3, 4, 3, 4, 2, 4, 2, 3, 4, 2, 4, 2, 3, 4, 3, 1, 3, 2, 3, 4, 3, 2, 4, 3, 4, 5, 6, 5, 4, 2, 1, 3, 2, 3, 1, 3, 1, 2, 3, 2, 4, 3, 2, 3, 4, 3, 4, 2, 1, 3, 4, 2, 4, 3, 1, 2, 3, 4, 2, 1, 2, 3, 4, 2, 3, 2, 4, 3, 1, 3, 2, 1, 3, 2, 1, 3, 1, 8, 7, 6, 8, 6, 7, 5, 7, 5, 4, 3, 1, 3, 2, 1, 8, 7, 8, 1, 2, 1, 8, 6, 7, 5, 4, 3, 2, 4, 3, 1, 8, 6, 5, 6, 8, 7, 5, 4, 2, 1, 8, 1, 2, 1, 8, 1, 3, 1, 3, 4, 2, 1, 3, 4, 2, 1, 2, 1, 3, 1, 8, 6, 7, 8, 6, 7, 8, 6, 8, 1, 8, 6, 8, 6, 5, 4, 5, 7, 6, 8, 6, 7, 5, 7, 6, 5, 4, 5, 4, 2, 1, 3, 4, 5, 4, 2, 3, 4, 2, 1, 8, 1, 3, 4, 5, 4, 2, 4, 2, 1, 2, 1, 3, 4, 3, 2, 3, 1, 2, 3, 1, 2, 4, 2, 1, 8, 7, 6, 8, 6, 8, 1, 8, 6, 8, 1, 2, 1, 3, 4, 3, 4, 5, 7, 8, 7, 6, 5, 6, 8, 1, 8, 7, 5, 6, 8, 6, 8, 1, 3, 2, 1, 8, 1, 8, 7, 8, 6, 5, 7, 5, 4, 3, 1, 2, 4, 5, 6, 5, 6, 5, 4, 3, 4, 2, 4, 3, 1, 3, 1, 8, 6, 8, 1, 2, 1, 3, 4, 3, 4, 3, 4, 2, 3, 1, 3, 4, 5, 4, 3, 2, 3, 4, 2, 3, 2, 4, 5, 6, 8, 7, 8, 7, 8, 7, 5, 7, 6, 8, 1, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 6, 8, 1, 8, 1, 2, 3, 2, 4, 5, 7, 5, 7, 5, 7, 5, 6, 8, 7, 5, 4, 5, 7, 8, 6, 5}
			} else if ss.runnum == 9 {
				seqid = [900]int{4, 2, 4, 3, 4, 2, 1, 3, 4, 2, 4, 5, 6, 5, 4, 3, 2, 1, 8, 7, 5, 7, 5, 6, 5, 7, 6, 5, 7, 6, 7, 6, 7, 6, 8, 7, 5, 6, 8, 1, 3, 4, 2, 3, 4, 3, 2, 1, 2, 4, 5, 6, 5, 7, 8, 7, 5, 7, 8, 1, 2, 4, 3, 4, 5, 6, 7, 6, 7, 6, 5, 7, 6, 5, 6, 8, 1, 3, 1, 8, 1, 8, 1, 2, 1, 3, 2, 1, 8, 6, 7, 6, 8, 1, 2, 4, 5, 4, 5, 6, 7, 6, 7, 5, 6, 8, 1, 2, 1, 2, 3, 2, 4, 3, 2, 1, 3, 1, 3, 4, 3, 4, 3, 2, 1, 3, 1, 8, 1, 8, 1, 8, 6, 8, 6, 7, 8, 6, 7, 6, 5, 7, 5, 4, 5, 7, 6, 8, 6, 5, 6, 7, 6, 8, 6, 5, 6, 7, 5, 4, 5, 7, 6, 5, 7, 5, 4, 5, 6, 8, 7, 6, 7, 6, 7, 5, 7, 5, 6, 7, 8, 6, 8, 1, 2, 1, 8, 6, 5, 4, 2, 3, 4, 2, 1, 8, 7, 8, 1, 3, 1, 2, 4, 5, 7, 5, 4, 2, 4, 2, 4, 5, 6, 5, 6, 8, 1, 2, 3, 2, 4, 2, 3, 1, 3, 1, 2, 4, 5, 4, 2, 1, 8, 6, 7, 6, 8, 1, 3, 4, 2, 1, 3, 1, 2, 1, 8, 7, 8, 1, 3, 1, 3, 1, 8, 1, 2, 3, 4, 2, 4, 2, 3, 2, 1, 2, 1, 2, 4, 5, 6, 5, 6, 7, 6, 5, 4, 3, 4, 2, 4, 5, 6, 8, 1, 8, 7, 5, 7, 5, 7, 5, 7, 8, 7, 5, 6, 5, 4, 3, 1, 2, 1, 2, 3, 4, 5, 7, 5, 4, 2, 4, 2, 1, 8, 6, 5, 4, 3, 4, 2, 4, 3, 1, 2, 4, 3, 2, 4, 5, 6, 8, 7, 6, 7, 6, 7, 6, 8, 1, 8, 6, 8, 7, 8, 6, 8, 7, 8, 6, 5, 6, 8, 7, 6, 7, 6, 7, 6, 8, 1, 3, 2, 4, 2, 4, 2, 1, 2, 4, 2, 4, 5, 6, 7, 5, 7, 5, 4, 3, 2, 1, 3, 4, 2, 4, 5, 4, 3, 2, 4, 3, 1, 8, 1, 2, 4, 5, 7, 6, 7, 5, 7, 5, 7, 5, 6, 8, 7, 6, 5, 7, 6, 5, 4, 3, 2, 3, 4, 3, 4, 2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 4, 2, 4, 5, 4, 2, 1, 2, 1, 8, 6, 7, 8, 7, 6, 5, 4, 5, 6, 7, 6, 7, 6, 5, 7, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 1, 3, 4, 2, 3, 1, 3, 4, 2, 1, 8, 1, 2, 1, 8, 6, 8, 7, 5, 6, 8, 7, 8, 7, 8, 7, 6, 5, 6, 7, 6, 5, 7, 5, 6, 8, 7, 5, 4, 3, 2, 1, 8, 6, 5, 7, 5, 6, 5, 4, 5, 6, 7, 6, 8, 1, 3, 4, 3, 4, 2, 1, 8, 1, 3, 2, 1, 2, 3, 1, 8, 6, 5, 6, 7, 8, 6, 7, 5, 4, 3, 2, 4, 3, 2, 3, 1, 3, 4, 2, 4, 2, 1, 3, 1, 2, 3, 1, 3, 4, 5, 6, 8, 7, 5, 7, 6, 7, 5, 7, 8, 6, 5, 6, 8, 1, 3, 2, 3, 2, 1, 3, 4, 3, 2, 4, 2, 1, 2, 1, 3, 4, 3, 4, 3, 1, 8, 7, 5, 4, 3, 2, 1, 2, 4, 5, 7, 8, 1, 8, 1, 3, 4, 5, 7, 8, 6, 7, 8, 6, 8, 1, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 1, 8, 6, 8, 7, 8, 7, 8, 6, 8, 1, 3, 1, 8, 6, 5, 6, 8, 1, 8, 6, 7, 5, 4, 3, 4, 5, 4, 2, 4, 2, 4, 5, 7, 5, 6, 7, 8, 6, 8, 7, 8, 7, 6, 7, 8, 7, 5, 6, 7, 8, 6, 7, 5, 4, 3, 1, 3, 1, 8, 7, 8, 6, 8, 6, 7, 8, 6, 7, 8, 6, 8, 7, 6, 7, 8, 1, 3, 1, 8, 1, 3, 1, 8, 6, 7, 6, 8, 1, 8, 1, 2, 4, 2, 1, 3, 4, 5, 7, 5, 7, 6, 5, 7, 6, 7, 6, 7, 8, 6, 8, 7, 5, 4, 5, 7, 8, 1, 3, 1, 2, 4, 2, 3, 2, 1, 2, 4, 5, 4, 3, 4, 3, 2, 4, 2, 1, 8, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 4, 5, 6, 8, 6, 5, 4, 2, 4, 5, 4, 5, 7, 6, 8, 7, 5, 7, 6, 5, 7, 6, 8, 1, 3, 2, 1, 2, 4, 5, 4, 2, 1, 2, 4, 2, 3, 2, 3, 4, 5, 7, 8, 1, 2, 3, 2, 1, 3, 4, 2, 4, 5, 6, 5, 6, 8, 1, 2, 4, 2, 4, 2, 1, 8, 6, 8, 6, 7, 5, 4, 2, 1, 8, 7, 6, 7, 6, 5, 6, 7, 6, 5, 4, 2, 4, 2, 1, 8, 1, 8, 7, 5, 6, 8, 7, 6, 5, 6, 5, 4, 5, 6, 5, 6, 8, 1, 2, 4, 5, 7, 6, 5, 7}
			} else if ss.runnum == 10 {
				seqid = [900]int{3, 1, 3, 1, 3, 4, 3, 4, 2, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6, 8, 6, 8, 6, 7, 5, 6, 7, 6, 7, 5, 4, 5, 6, 8, 7, 5, 6, 7, 8, 6, 7, 5, 7, 6, 7, 6, 5, 6, 5, 7, 8, 6, 7, 6, 5, 7, 8, 7, 8, 7, 6, 8, 7, 5, 7, 8, 7, 8, 1, 8, 7, 6, 5, 6, 7, 8, 6, 7, 5, 4, 3, 4, 2, 3, 1, 3, 4, 5, 7, 6, 5, 6, 7, 5, 7, 6, 8, 7, 8, 1, 8, 6, 7, 6, 5, 7, 8, 6, 5, 4, 3, 4, 3, 1, 3, 2, 3, 1, 8, 1, 3, 1, 2, 3, 2, 3, 1, 2, 3, 4, 3, 4, 5, 4, 5, 4, 3, 1, 8, 1, 2, 3, 1, 3, 1, 8, 6, 7, 6, 7, 8, 1, 2, 4, 3, 4, 3, 4, 2, 3, 4, 5, 6, 5, 7, 6, 7, 5, 4, 5, 4, 5, 4, 2, 1, 3, 1, 2, 1, 2, 4, 5, 6, 7, 8, 7, 6, 8, 7, 8, 1, 8, 6, 7, 8, 6, 7, 6, 8, 7, 5, 6, 7, 6, 7, 5, 4, 5, 7, 5, 4, 3, 2, 1, 8, 6, 5, 4, 5, 7, 5, 4, 5, 7, 6, 5, 6, 8, 6, 7, 8, 6, 7, 6, 7, 8, 1, 8, 6, 8, 1, 8, 1, 3, 4, 3, 4, 3, 4, 2, 1, 2, 4, 3, 2, 3, 4, 3, 2, 3, 2, 3, 4, 5, 4, 3, 1, 2, 4, 5, 6, 5, 7, 5, 6, 7, 6, 8, 1, 8, 1, 2, 3, 4, 2, 1, 8, 6, 5, 6, 5, 7, 8, 6, 5, 6, 7, 8, 6, 7, 8, 7, 5, 7, 8, 6, 5, 4, 3, 2, 1, 8, 7, 8, 1, 3, 1, 3, 4, 2, 4, 5, 6, 8, 6, 8, 7, 8, 1, 3, 1, 2, 1, 3, 4, 5, 4, 5, 7, 8, 6, 7, 8, 6, 5, 7, 6, 8, 7, 8, 7, 6, 8, 1, 2, 3, 1, 8, 6, 5, 6, 5, 4, 5, 7, 8, 6, 5, 4, 5, 4, 2, 3, 1, 3, 4, 2, 3, 2, 3, 1, 3, 2, 3, 1, 3, 4, 3, 4, 3, 2, 3, 2, 4, 2, 4, 5, 7, 6, 7, 8, 7, 8, 1, 3, 2, 4, 5, 6, 5, 4, 2, 1, 3, 2, 4, 5, 7, 8, 7, 8, 7, 6, 7, 5, 4, 5, 4, 5, 4, 2, 4, 2, 3, 1, 2, 1, 8, 6, 5, 7, 5, 6, 7, 6, 5, 7, 6, 8, 1, 3, 4, 3, 2, 3, 4, 5, 7, 5, 4, 2, 1, 3, 4, 3, 4, 3, 4, 2, 3, 4, 2, 3, 4, 3, 4, 2, 4, 3, 4, 2, 3, 2, 3, 4, 2, 1, 3, 4, 2, 3, 2, 4, 3, 4, 2, 1, 8, 6, 7, 8, 1, 8, 1, 2, 1, 8, 1, 3, 1, 8, 1, 3, 1, 3, 1, 2, 3, 1, 3, 2, 4, 5, 4, 3, 1, 2, 4, 3, 1, 8, 1, 8, 7, 6, 7, 6, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 1, 3, 2, 3, 1, 2, 3, 4, 2, 3, 2, 4, 5, 4, 2, 4, 5, 6, 8, 6, 5, 6, 5, 4, 5, 4, 5, 4, 5, 6, 7, 8, 7, 5, 7, 8, 7, 6, 7, 8, 6, 8, 6, 7, 5, 4, 2, 3, 4, 2, 4, 3, 2, 3, 4, 2, 3, 1, 8, 6, 8, 6, 5, 4, 5, 4, 2, 4, 2, 1, 3, 4, 3, 1, 3, 4, 2, 3, 1, 3, 4, 2, 3, 1, 8, 1, 8, 7, 8, 6, 7, 6, 7, 6, 8, 1, 3, 1, 8, 7, 6, 7, 8, 7, 5, 6, 5, 6, 8, 1, 8, 7, 5, 6, 5, 6, 8, 7, 8, 6, 5, 4, 3, 4, 5, 7, 6, 8, 7, 8, 6, 5, 4, 2, 4, 5, 4, 2, 4, 3, 4, 5, 7, 6, 5, 4, 3, 1, 3, 1, 2, 4, 3, 2, 3, 2, 4, 5, 4, 2, 4, 3, 1, 2, 4, 2, 4, 2, 1, 8, 1, 3, 1, 3, 2, 3, 1, 2, 4, 5, 7, 6, 8, 6, 5, 6, 8, 1, 2, 4, 2, 4, 5, 7, 6, 7, 6, 5, 7, 5, 7, 8, 1, 3, 1, 2, 1, 2, 4, 5, 7, 5, 7, 6, 8, 7, 5, 7, 8, 6, 5, 4, 2, 3, 1, 8, 1, 3, 2, 1, 2, 3, 4, 5, 7, 6, 7, 8, 7, 5, 6, 5, 6, 8, 7, 5, 7, 8, 1, 2, 3, 1, 3, 4, 5, 4, 2, 1, 2, 4, 5, 6, 7, 6, 8, 1, 3, 4, 5, 7, 5, 7, 6, 7, 6, 7, 5, 4, 2, 4, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2, 4, 5, 4, 2, 4, 2, 1, 2, 4, 5, 7, 6, 5, 7, 6, 5, 4, 5, 4, 2, 1, 3, 2, 4, 2, 3, 1, 8, 1, 8, 7, 6, 5, 6, 7, 8, 1, 2, 1, 2, 3, 2, 3, 1, 8, 7, 5, 7, 5, 4, 3, 4, 5, 4, 5, 6, 7}
			} else if ss.runnum == 11 {
				seqid = [900]int{1, 8, 1, 8, 7, 6, 7, 6, 7, 6, 8, 6, 8, 1, 3, 1, 8, 7, 8, 1, 3, 1, 8, 6, 8, 1, 8, 1, 2, 3, 4, 2, 3, 4, 2, 1, 3, 4, 5, 7, 5, 4, 3, 2, 4, 3, 2, 3, 2, 3, 1, 8, 6, 7, 6, 7, 6, 7, 8, 1, 3, 2, 4, 3, 2, 1, 3, 4, 5, 6, 8, 6, 7, 6, 5, 4, 3, 2, 1, 2, 4, 5, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 4, 5, 6, 8, 1, 3, 2, 1, 2, 1, 3, 1, 8, 1, 3, 1, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2, 1, 2, 1, 2, 1, 8, 7, 6, 5, 6, 5, 7, 6, 5, 7, 6, 7, 5, 4, 5, 4, 2, 1, 2, 4, 2, 4, 5, 7, 8, 6, 8, 6, 8, 1, 2, 4, 5, 4, 2, 1, 3, 4, 5, 6, 5, 6, 7, 6, 8, 1, 2, 1, 2, 1, 2, 4, 5, 4, 5, 4, 5, 6, 5, 4, 5, 4, 2, 3, 2, 3, 4, 2, 1, 2, 3, 1, 3, 1, 2, 4, 3, 4, 5, 4, 5, 7, 8, 1, 3, 1, 8, 1, 2, 4, 3, 4, 5, 6, 5, 4, 3, 4, 5, 4, 5, 4, 5, 6, 5, 6, 7, 6, 5, 4, 2, 1, 2, 4, 5, 7, 6, 5, 6, 8, 7, 6, 8, 6, 7, 6, 7, 8, 6, 7, 6, 5, 4, 5, 7, 6, 8, 6, 7, 5, 4, 2, 4, 5, 7, 5, 4, 5, 4, 3, 2, 3, 4, 3, 4, 5, 6, 8, 6, 7, 6, 5, 7, 6, 7, 8, 7, 8, 1, 3, 2, 4, 5, 6, 7, 8, 1, 8, 6, 5, 7, 8, 7, 6, 8, 1, 3, 4, 5, 7, 8, 7, 5, 7, 8, 1, 8, 7, 8, 1, 3, 2, 3, 2, 1, 2, 4, 3, 1, 2, 3, 1, 3, 4, 5, 4, 3, 1, 8, 6, 5, 4, 5, 7, 8, 1, 8, 7, 5, 4, 2, 1, 8, 1, 2, 3, 1, 3, 2, 1, 8, 1, 2, 1, 3, 1, 2, 4, 3, 2, 4, 5, 6, 8, 6, 8, 1, 3, 1, 2, 4, 5, 7, 5, 4, 3, 4, 2, 3, 1, 3, 2, 3, 1, 3, 2, 4, 2, 4, 3, 4, 3, 1, 2, 1, 3, 2, 1, 8, 1, 2, 3, 1, 8, 6, 8, 7, 6, 7, 6, 5, 4, 5, 6, 7, 8, 1, 2, 1, 8, 6, 8, 7, 5, 7, 6, 8, 7, 8, 1, 2, 1, 8, 1, 2, 1, 2, 3, 1, 8, 7, 5, 6, 7, 6, 5, 4, 3, 1, 2, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 4, 2, 3, 1, 3, 4, 2, 1, 3, 1, 8, 6, 5, 7, 5, 4, 5, 4, 2, 4, 2, 3, 2, 1, 2, 1, 8, 7, 5, 6, 8, 7, 5, 7, 6, 7, 6, 8, 6, 8, 1, 2, 3, 2, 4, 5, 6, 7, 8, 6, 8, 7, 6, 5, 6, 8, 6, 5, 4, 3, 1, 3, 1, 2, 4, 2, 4, 5, 6, 5, 6, 8, 1, 2, 4, 2, 4, 2, 1, 8, 6, 7, 6, 8, 1, 3, 2, 1, 2, 4, 2, 4, 3, 2, 3, 1, 2, 1, 2, 4, 2, 1, 3, 4, 2, 1, 2, 1, 2, 4, 2, 3, 1, 3, 2, 4, 5, 4, 3, 4, 5, 7, 5, 4, 2, 4, 3, 1, 3, 2, 1, 3, 2, 1, 8, 1, 3, 4, 5, 6, 7, 6, 7, 8, 7, 6, 8, 7, 8, 7, 5, 6, 7, 5, 6, 8, 1, 8, 1, 2, 1, 8, 1, 2, 1, 2, 3, 1, 2, 3, 2, 1, 2, 4, 5, 4, 3, 1, 2, 1, 2, 3, 2, 1, 8, 6, 7, 8, 6, 5, 4, 3, 2, 3, 1, 2, 4, 2, 4, 3, 4, 3, 4, 2, 3, 2, 4, 5, 6, 5, 4, 5, 6, 5, 4, 3, 1, 2, 1, 2, 3, 4, 3, 4, 3, 1, 2, 1, 3, 4, 5, 6, 7, 8, 6, 8, 7, 5, 4, 3, 1, 8, 1, 3, 4, 2, 1, 2, 4, 2, 4, 3, 2, 1, 2, 1, 8, 6, 7, 6, 5, 7, 8, 1, 8, 7, 5, 7, 6, 7, 8, 7, 5, 7, 5, 4, 3, 2, 4, 2, 1, 8, 7, 6, 7, 5, 4, 5, 6, 7, 5, 7, 5, 6, 8, 7, 6, 7, 8, 6, 8, 7, 6, 8, 6, 7, 8, 6, 5, 6, 7, 5, 7, 6, 8, 6, 8, 7, 8, 6, 7, 5, 4, 2, 1, 8, 6, 5, 4, 3, 1, 2, 4, 5, 7, 6, 7, 6, 7, 8, 1, 2, 3, 4, 2, 1, 3, 2, 1, 8, 6, 8, 1, 2, 3, 2, 1, 2, 3, 1, 8, 1, 3, 1, 3, 4, 2, 1, 2, 4, 2, 4, 5, 4, 3, 1, 3, 2, 1, 3, 2, 4, 2, 3, 2, 3, 4, 2, 3, 1, 2, 1, 3, 4, 5, 6, 8, 1, 2, 4, 5, 7, 6, 5, 4, 5, 7, 8, 7, 8, 6, 5, 6, 5, 6, 7, 8, 7, 6, 7, 5, 4, 3, 2, 4, 2}
			} else if ss.runnum == 12 {
				seqid = [900]int{4, 2, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 6, 7, 8, 1, 8, 7, 8, 7, 6, 7, 5, 4, 5, 6, 5, 7, 5, 4, 3, 4, 5, 7, 5, 4, 3, 2, 4, 3, 2, 4, 3, 1, 2, 4, 5, 7, 6, 7, 8, 7, 8, 6, 8, 7, 6, 7, 5, 6, 8, 7, 5, 4, 5, 4, 5, 7, 6, 7, 8, 1, 3, 1, 8, 7, 5, 6, 8, 1, 3, 4, 5, 6, 7, 6, 8, 1, 3, 1, 3, 2, 1, 8, 6, 7, 5, 7, 6, 5, 7, 6, 7, 6, 5, 4, 3, 4, 5, 4, 3, 4, 5, 6, 8, 6, 5, 6, 8, 1, 8, 1, 2, 1, 8, 7, 5, 6, 8, 1, 8, 6, 5, 4, 5, 4, 3, 1, 8, 1, 2, 3, 2, 3, 1, 8, 6, 5, 7, 5, 4, 2, 1, 8, 1, 2, 1, 8, 6, 5, 7, 5, 6, 7, 5, 4, 5, 6, 8, 1, 8, 7, 5, 7, 5, 6, 7, 8, 7, 8, 7, 5, 7, 5, 4, 2, 4, 2, 4, 5, 4, 5, 4, 5, 6, 5, 6, 5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3, 4, 2, 1, 8, 6, 5, 6, 8, 1, 8, 1, 3, 4, 5, 4, 2, 3, 4, 2, 3, 2, 1, 8, 1, 2, 4, 5, 7, 8, 1, 3, 2, 3, 2, 4, 3, 2, 3, 1, 8, 1, 8, 7, 8, 7, 6, 5, 7, 6, 7, 8, 6, 8, 6, 5, 7, 8, 1, 8, 7, 8, 6, 5, 7, 6, 5, 4, 3, 4, 2, 1, 8, 7, 8, 1, 8, 1, 8, 7, 6, 7, 5, 6, 5, 6, 8, 7, 6, 7, 8, 6, 8, 1, 3, 4, 3, 1, 8, 6, 8, 7, 6, 8, 7, 8, 7, 8, 7, 5, 7, 5, 4, 2, 3, 2, 1, 2, 3, 1, 8, 6, 7, 8, 7, 6, 7, 8, 1, 8, 7, 5, 6, 5, 7, 8, 1, 3, 2, 1, 8, 1, 3, 4, 2, 1, 2, 3, 1, 3, 4, 5, 7, 5, 7, 8, 1, 2, 1, 8, 6, 5, 6, 7, 5, 4, 5, 4, 5, 6, 8, 1, 3, 1, 3, 2, 4, 3, 2, 3, 4, 2, 3, 4, 3, 1, 2, 1, 8, 1, 2, 1, 8, 7, 8, 7, 8, 1, 3, 2, 4, 3, 4, 2, 4, 3, 2, 1, 2, 1, 3, 1, 3, 4, 3, 4, 3, 1, 8, 7, 6, 7, 5, 4, 2, 4, 5, 4, 2, 1, 2, 1, 2, 1, 3, 1, 2, 1, 8, 1, 8, 6, 7, 8, 6, 7, 5, 7, 6, 8, 7, 5, 7, 6, 5, 4, 3, 4, 5, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 7, 6, 8, 1, 3, 2, 1, 3, 2, 3, 1, 3, 4, 3, 4, 3, 4, 2, 1, 2, 3, 4, 5, 6, 7, 6, 8, 6, 5, 6, 8, 7, 5, 7, 6, 8, 7, 5, 6, 8, 1, 3, 4, 5, 6, 8, 1, 3, 1, 2, 4, 5, 6, 5, 6, 8, 6, 5, 7, 8, 7, 8, 6, 7, 6, 7, 6, 5, 4, 5, 7, 8, 7, 5, 4, 3, 1, 8, 6, 7, 6, 5, 4, 5, 7, 5, 6, 7, 8, 7, 8, 6, 7, 6, 7, 5, 6, 8, 7, 5, 6, 8, 6, 7, 6, 5, 6, 8, 1, 3, 2, 1, 8, 7, 6, 8, 7, 8, 1, 3, 1, 8, 7, 5, 4, 2, 3, 2, 4, 2, 1, 3, 1, 8, 7, 8, 1, 8, 6, 5, 7, 6, 7, 6, 8, 1, 3, 4, 2, 4, 3, 2, 3, 2, 1, 3, 2, 4, 5, 7, 8, 7, 6, 5, 6, 7, 6, 8, 1, 3, 4, 3, 4, 3, 2, 3, 2, 3, 1, 3, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 2, 4, 3, 2, 3, 4, 2, 3, 2, 1, 8, 7, 5, 7, 5, 4, 5, 6, 8, 1, 3, 1, 8, 7, 6, 8, 7, 5, 6, 8, 1, 8, 7, 6, 5, 7, 8, 6, 8, 7, 5, 7, 6, 5, 7, 8, 6, 8, 6, 8, 1, 3, 2, 3, 4, 2, 3, 4, 5, 6, 8, 1, 2, 3, 1, 3, 1, 8, 6, 5, 7, 8, 7, 8, 7, 6, 8, 6, 5, 4, 3, 2, 1, 8, 1, 2, 4, 5, 4, 5, 6, 7, 6, 5, 7, 6, 8, 6, 8, 1, 3, 4, 5, 7, 5, 7, 6, 5, 7, 8, 6, 5, 6, 8, 7, 6, 5, 7, 5, 7, 5, 6, 7, 8, 7, 6, 7, 6, 7, 8, 6, 7, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 5, 4, 2, 3, 2, 1, 3, 2, 3, 4, 3, 4, 3, 4, 2, 4, 2, 4, 2, 4, 2, 3, 4, 5, 6, 8, 7, 8, 7, 8, 1, 3, 1, 2, 3, 1, 3, 2, 4, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 6, 8, 6, 8, 1, 2, 1, 3, 1, 3, 2, 3, 4, 3, 1, 2, 3, 1, 3, 4, 2, 3, 4, 3, 4, 5, 6, 8, 1, 3, 4, 3, 1, 2, 4, 5, 6, 5, 7, 5, 6}
			} else if ss.runnum == 13 {
				seqid = [900]int{2, 1, 2, 4, 3, 1, 3, 2, 4, 3, 2, 4, 2, 1, 8, 7, 5, 6, 5, 7, 8, 7, 5, 4, 3, 2, 3, 4, 3, 2, 4, 5, 4, 3, 4, 3, 2, 4, 5, 7, 5, 6, 8, 6, 7, 5, 6, 5, 4, 2, 4, 5, 4, 3, 2, 4, 5, 7, 5, 4, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 4, 2, 1, 3, 4, 5, 6, 8, 1, 8, 1, 2, 1, 8, 6, 8, 6, 7, 6, 8, 6, 5, 4, 3, 4, 3, 4, 3, 2, 1, 3, 2, 3, 2, 3, 1, 8, 6, 7, 5, 4, 2, 1, 2, 3, 2, 4, 5, 6, 8, 6, 8, 1, 2, 3, 4, 5, 6, 5, 4, 2, 1, 2, 3, 2, 1, 3, 4, 5, 6, 5, 7, 8, 1, 2, 1, 3, 4, 3, 2, 1, 8, 7, 8, 6, 5, 6, 8, 1, 8, 1, 2, 3, 1, 3, 2, 1, 3, 1, 8, 7, 6, 7, 5, 7, 6, 7, 8, 6, 5, 4, 2, 1, 8, 7, 6, 5, 7, 5, 6, 5, 4, 3, 4, 2, 4, 3, 2, 3, 2, 3, 4, 5, 6, 5, 6, 8, 6, 8, 6, 7, 5, 4, 2, 1, 8, 1, 8, 1, 2, 1, 8, 1, 8, 1, 2, 3, 1, 8, 6, 5, 7, 8, 7, 5, 6, 5, 4, 2, 1, 2, 3, 2, 4, 5, 4, 2, 4, 3, 4, 2, 3, 1, 3, 4, 2, 3, 1, 8, 7, 8, 7, 6, 5, 6, 5, 6, 5, 7, 8, 1, 8, 1, 3, 2, 4, 2, 1, 8, 1, 8, 1, 8, 1, 8, 7, 5, 7, 6, 5, 7, 8, 6, 5, 6, 5, 7, 8, 7, 8, 1, 2, 3, 2, 4, 2, 3, 2, 4, 5, 7, 6, 5, 4, 3, 2, 4, 2, 4, 3, 4, 5, 7, 5, 4, 2, 3, 4, 2, 4, 3, 1, 2, 1, 8, 6, 5, 4, 3, 1, 8, 6, 7, 6, 5, 6, 5, 4, 5, 6, 5, 6, 5, 7, 5, 4, 2, 4, 5, 4, 5, 4, 2, 4, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 8, 7, 5, 6, 7, 5, 6, 5, 6, 7, 6, 7, 5, 7, 6, 8, 7, 6, 5, 4, 3, 1, 3, 4, 5, 6, 8, 1, 8, 6, 7, 5, 4, 2, 4, 2, 1, 3, 1, 8, 7, 8, 7, 8, 6, 8, 1, 3, 2, 4, 3, 1, 2, 3, 2, 3, 4, 2, 3, 4, 2, 3, 2, 4, 5, 4, 2, 4, 5, 7, 6, 5, 4, 2, 4, 5, 6, 7, 5, 6, 5, 6, 8, 1, 8, 1, 2, 4, 5, 6, 7, 8, 1, 2, 4, 2, 1, 8, 6, 8, 6, 5, 4, 3, 2, 4, 5, 7, 5, 4, 3, 4, 2, 4, 3, 2, 3, 1, 3, 1, 8, 6, 5, 4, 2, 4, 5, 7, 5, 6, 7, 5, 7, 6, 7, 8, 1, 3, 4, 3, 4, 2, 4, 5, 7, 6, 7, 6, 5, 4, 2, 3, 1, 2, 1, 2, 1, 3, 1, 2, 4, 2, 1, 8, 7, 8, 1, 8, 6, 8, 1, 2, 4, 2, 4, 3, 4, 3, 2, 4, 5, 4, 5, 7, 5, 6, 7, 8, 7, 8, 7, 6, 7, 8, 6, 7, 8, 1, 2, 4, 5, 4, 5, 6, 5, 4, 5, 6, 8, 6, 5, 7, 5, 4, 3, 4, 2, 1, 3, 2, 3, 4, 2, 4, 3, 2, 4, 2, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1, 8, 7, 6, 8, 1, 2, 4, 2, 1, 3, 4, 3, 2, 1, 3, 2, 3, 4, 3, 4, 3, 4, 2, 3, 1, 3, 4, 2, 4, 3, 1, 3, 2, 3, 4, 5, 4, 3, 4, 3, 2, 1, 2, 4, 2, 1, 8, 1, 8, 1, 8, 6, 8, 1, 3, 1, 8, 1, 2, 4, 3, 1, 2, 4, 2, 3, 4, 2, 3, 4, 5, 4, 5, 6, 8, 1, 2, 3, 2, 4, 2, 3, 1, 2, 4, 3, 4, 5, 7, 6, 5, 6, 5, 6, 8, 6, 8, 1, 2, 3, 2, 1, 3, 2, 3, 4, 2, 4, 5, 6, 5, 4, 5, 4, 5, 7, 8, 7, 8, 7, 8, 7, 8, 6, 5, 4, 2, 4, 5, 7, 6, 8, 7, 5, 7, 5, 7, 5, 4, 3, 4, 3, 2, 4, 2, 4, 3, 2, 4, 5, 6, 8, 7, 6, 5, 6, 5, 7, 6, 7, 8, 6, 8, 6, 8, 6, 8, 7, 6, 5, 4, 3, 4, 5, 7, 6, 5, 7, 5, 6, 7, 8, 7, 8, 1, 8, 7, 6, 7, 6, 7, 6, 5, 7, 5, 4, 5, 4, 3, 2, 3, 1, 8, 1, 8, 7, 6, 7, 8, 1, 2, 1, 2, 4, 5, 7, 6, 8, 6, 5, 7, 6, 7, 5, 7, 5, 6, 5, 4, 5, 4, 2, 1, 8, 1, 8, 6, 8, 7, 5, 4, 3, 1, 8, 1, 3, 4, 3, 4, 3, 2, 1, 3, 4, 3, 2, 1, 2, 4, 2, 4, 2, 3, 1, 3, 2, 3, 1, 8, 7, 8, 6, 8, 6, 5, 7, 5, 7, 8, 1, 2, 1, 2}
			} else if ss.runnum == 14 {
				seqid = [900]int{7, 5, 6, 7, 6, 5, 6, 7, 8, 6, 5, 6, 8, 1, 2, 1, 2, 4, 2, 4, 2, 3, 1, 8, 6, 5, 7, 6, 7, 8, 6, 5, 7, 8, 6, 8, 1, 8, 6, 5, 6, 5, 4, 2, 1, 8, 1, 2, 1, 2, 4, 3, 2, 1, 3, 1, 8, 7, 5, 4, 3, 1, 3, 2, 3, 1, 3, 1, 8, 1, 2, 4, 3, 4, 3, 2, 1, 2, 4, 5, 6, 7, 6, 8, 6, 5, 4, 2, 1, 8, 1, 2, 3, 1, 2, 4, 5, 6, 7, 6, 7, 5, 6, 5, 4, 5, 7, 8, 7, 5, 6, 8, 1, 2, 3, 1, 8, 6, 8, 7, 6, 7, 8, 6, 8, 6, 5, 7, 8, 6, 7, 6, 8, 1, 2, 1, 3, 2, 3, 4, 2, 1, 8, 6, 8, 7, 8, 1, 2, 3, 4, 3, 2, 4, 3, 2, 1, 3, 4, 3, 1, 3, 2, 4, 2, 4, 5, 7, 6, 7, 8, 7, 8, 7, 8, 1, 8, 1, 3, 2, 4, 5, 4, 2, 4, 5, 7, 5, 4, 3, 2, 1, 8, 1, 8, 7, 6, 7, 6, 5, 6, 8, 7, 8, 7, 6, 7, 5, 7, 5, 6, 7, 5, 7, 8, 6, 8, 7, 5, 6, 8, 6, 5, 7, 5, 4, 5, 6, 7, 6, 8, 6, 8, 6, 7, 6, 5, 7, 6, 5, 7, 6, 7, 5, 7, 5, 7, 6, 8, 7, 8, 7, 6, 8, 6, 8, 6, 7, 8, 7, 8, 7, 6, 7, 8, 1, 8, 1, 3, 1, 2, 3, 2, 3, 4, 3, 2, 3, 2, 1, 3, 4, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 2, 4, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 7, 6, 8, 7, 5, 6, 8, 6, 5, 6, 8, 1, 8, 1, 3, 2, 1, 8, 7, 6, 7, 5, 7, 8, 7, 6, 8, 7, 8, 7, 5, 6, 7, 8, 1, 3, 2, 1, 2, 1, 3, 4, 5, 4, 5, 4, 2, 3, 1, 2, 3, 1, 2, 3, 4, 2, 4, 3, 1, 8, 6, 5, 6, 7, 5, 7, 8, 6, 7, 8, 7, 5, 6, 7, 8, 7, 8, 6, 5, 7, 6, 7, 8, 6, 5, 6, 7, 5, 4, 5, 7, 6, 8, 7, 8, 6, 8, 1, 2, 1, 8, 6, 7, 5, 6, 8, 7, 8, 6, 7, 5, 6, 7, 8, 7, 6, 7, 6, 8, 6, 5, 7, 5, 6, 7, 6, 5, 4, 3, 4, 3, 4, 5, 4, 3, 2, 3, 2, 3, 4, 2, 3, 1, 2, 3, 1, 8, 7, 5, 7, 5, 4, 5, 7, 8, 1, 3, 4, 2, 4, 5, 4, 5, 4, 5, 4, 2, 4, 2, 3, 1, 8, 7, 8, 6, 8, 7, 6, 8, 6, 7, 5, 4, 2, 1, 3, 2, 4, 5, 7, 6, 7, 6, 7, 6, 8, 6, 7, 6, 5, 4, 3, 4, 2, 4, 3, 1, 3, 4, 5, 7, 6, 8, 7, 5, 4, 2, 4, 2, 1, 8, 6, 7, 6, 8, 7, 6, 5, 7, 8, 6, 8, 6, 7, 5, 4, 3, 1, 8, 1, 2, 4, 5, 6, 8, 7, 5, 4, 2, 1, 8, 6, 8, 7, 5, 7, 5, 7, 8, 1, 3, 1, 8, 1, 3, 4, 5, 6, 5, 6, 8, 1, 8, 1, 3, 4, 3, 4, 3, 1, 2, 3, 2, 4, 2, 4, 5, 6, 7, 8, 7, 5, 4, 5, 4, 3, 4, 5, 4, 2, 1, 3, 2, 4, 3, 1, 3, 2, 1, 2, 4, 3, 1, 3, 4, 2, 3, 4, 3, 2, 3, 4, 2, 4, 5, 7, 8, 1, 2, 4, 2, 3, 2, 1, 8, 7, 8, 7, 8, 6, 8, 1, 8, 7, 5, 6, 8, 6, 5, 6, 8, 1, 3, 2, 3, 2, 1, 3, 1, 8, 7, 8, 1, 3, 2, 4, 2, 1, 3, 4, 5, 4, 5, 4, 2, 4, 2, 1, 3, 2, 4, 5, 7, 6, 5, 4, 3, 1, 3, 2, 1, 8, 7, 5, 4, 2, 4, 3, 1, 2, 1, 3, 1, 3, 1, 3, 2, 1, 2, 3, 2, 1, 3, 2, 3, 1, 3, 2, 3, 2, 3, 2, 4, 2, 1, 2, 1, 8, 1, 8, 6, 8, 7, 6, 5, 7, 5, 7, 6, 5, 6, 5, 6, 8, 7, 6, 8, 1, 3, 1, 3, 1, 8, 6, 5, 4, 2, 1, 8, 6, 7, 5, 7, 5, 6, 7, 5, 6, 5, 4, 5, 4, 2, 3, 1, 3, 2, 3, 2, 3, 1, 3, 1, 3, 2, 1, 2, 4, 3, 1, 8, 6, 8, 6, 8, 1, 8, 1, 3, 2, 1, 2, 3, 4, 2, 3, 4, 5, 7, 5, 4, 5, 7, 8, 7, 8, 7, 5, 4, 5, 4, 5, 6, 5, 4, 2, 1, 3, 2, 4, 2, 3, 4, 3, 2, 3, 1, 8, 1, 2, 3, 2, 1, 8, 1, 3, 1, 2, 1, 2, 3, 1, 8, 7, 6, 8, 1, 8, 6, 5, 7, 5, 7, 6, 5, 6, 8, 1, 8, 1, 2, 4, 2, 4, 3, 1, 2, 1, 3, 4, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 2, 1, 2, 3}
			} else if ss.runnum == 15 {
				seqid = [900]int{4, 3, 2, 4, 3, 4, 2, 1, 3, 1, 3, 1, 8, 7, 5, 7, 8, 6, 5, 6, 7, 6, 8, 1, 3, 2, 3, 1, 2, 4, 3, 2, 3, 2, 4, 2, 4, 2, 3, 2, 3, 1, 2, 4, 3, 4, 2, 4, 3, 2, 1, 2, 3, 2, 3, 1, 3, 4, 3, 4, 5, 6, 8, 1, 2, 1, 2, 3, 2, 3, 4, 2, 3, 4, 5, 7, 8, 7, 8, 7, 6, 5, 6, 5, 7, 5, 7, 5, 4, 5, 4, 3, 1, 8, 6, 5, 7, 6, 5, 4, 2, 3, 4, 3, 2, 1, 8, 7, 5, 6, 7, 6, 8, 6, 5, 6, 8, 7, 8, 7, 5, 4, 5, 4, 5, 4, 3, 2, 3, 2, 4, 3, 4, 5, 7, 6, 7, 8, 7, 6, 5, 6, 8, 1, 2, 1, 3, 4, 2, 1, 8, 7, 5, 6, 7, 5, 6, 5, 6, 5, 7, 5, 7, 6, 7, 8, 6, 5, 6, 8, 6, 5, 4, 2, 1, 3, 4, 3, 1, 8, 6, 5, 6, 7, 5, 6, 8, 6, 7, 8, 6, 5, 7, 8, 6, 5, 7, 5, 6, 7, 6, 5, 6, 7, 8, 6, 8, 6, 5, 6, 8, 7, 8, 6, 7, 5, 4, 2, 4, 2, 4, 5, 6, 8, 1, 3, 1, 8, 6, 7, 5, 4, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8, 6, 5, 4, 3, 1, 8, 7, 5, 4, 3, 1, 8, 1, 8, 6, 8, 6, 5, 4, 2, 3, 4, 3, 2, 3, 4, 2, 1, 8, 1, 8, 7, 6, 7, 6, 7, 8, 6, 8, 7, 6, 7, 6, 5, 6, 7, 6, 8, 7, 5, 6, 8, 6, 5, 6, 7, 6, 5, 6, 5, 7, 6, 8, 6, 7, 6, 5, 4, 2, 4, 5, 6, 8, 6, 8, 7, 8, 6, 7, 8, 6, 5, 7, 5, 7, 8, 1, 8, 1, 2, 3, 4, 2, 1, 8, 7, 6, 5, 4, 3, 4, 2, 4, 5, 6, 7, 6, 7, 6, 5, 4, 3, 1, 8, 1, 2, 1, 8, 7, 6, 5, 6, 7, 8, 1, 3, 2, 1, 3, 2, 4, 5, 6, 7, 6, 5, 7, 8, 1, 2, 3, 2, 1, 8, 6, 8, 6, 7, 8, 7, 5, 4, 3, 1, 2, 4, 2, 4, 5, 4, 3, 1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 6, 7, 5, 6, 7, 6, 5, 7, 6, 5, 7, 8, 6, 7, 6, 8, 7, 8, 1, 3, 4, 5, 6, 8, 6, 7, 5, 4, 2, 4, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 2, 4, 3, 1, 3, 2, 3, 1, 2, 1, 8, 7, 5, 6, 7, 5, 7, 8, 7, 8, 1, 3, 4, 5, 6, 8, 1, 2, 3, 1, 2, 4, 3, 4, 2, 3, 1, 2, 1, 3, 4, 3, 4, 2, 4, 2, 4, 5, 6, 8, 7, 8, 6, 5, 7, 8, 7, 8, 7, 6, 5, 4, 2, 1, 8, 1, 2, 1, 2, 3, 2, 3, 2, 4, 2, 4, 2, 1, 3, 4, 3, 1, 3, 4, 2, 4, 3, 4, 2, 3, 1, 2, 3, 4, 3, 1, 2, 1, 8, 7, 8, 6, 7, 6, 7, 8, 1, 2, 4, 3, 1, 8, 7, 8, 6, 7, 8, 1, 8, 7, 5, 4, 5, 4, 5, 6, 8, 1, 8, 6, 8, 6, 5, 4, 5, 4, 3, 1, 2, 3, 4, 3, 4, 2, 1, 8, 7, 8, 6, 7, 8, 6, 7, 6, 7, 5, 7, 6, 5, 6, 5, 7, 8, 1, 8, 1, 8, 7, 8, 6, 5, 6, 8, 1, 2, 1, 8, 6, 7, 6, 7, 8, 1, 8, 7, 6, 5, 4, 3, 2, 3, 2, 4, 3, 1, 2, 3, 4, 3, 1, 3, 2, 4, 5, 6, 7, 8, 1, 8, 1, 8, 6, 7, 6, 7, 8, 1, 2, 4, 5, 4, 5, 4, 2, 4, 2, 4, 3, 2, 1, 3, 4, 3, 2, 3, 1, 2, 4, 5, 7, 8, 1, 3, 1, 8, 6, 7, 5, 6, 8, 6, 7, 8, 1, 2, 1, 2, 1, 3, 2, 1, 2, 4, 2, 1, 3, 2, 4, 2, 4, 2, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 8, 6, 8, 7, 6, 5, 4, 3, 1, 8, 1, 8, 1, 8, 1, 3, 1, 8, 1, 2, 1, 2, 4, 5, 4, 2, 3, 4, 3, 1, 8, 1, 3, 2, 3, 2, 3, 1, 2, 1, 2, 1, 8, 7, 6, 5, 7, 5, 7, 6, 5, 7, 8, 7, 8, 1, 8, 6, 7, 6, 8, 6, 5, 7, 6, 8, 6, 5, 7, 6, 8, 1, 2, 3, 4, 2, 1, 2, 1, 8, 6, 8, 7, 8, 6, 8, 6, 7, 5, 6, 7, 6, 7, 8, 7, 8, 1, 3, 1, 8, 7, 8, 1, 2, 1, 3, 4, 2, 3, 1, 2, 1, 2, 4, 2, 3, 4, 5, 4, 3, 1, 3, 4, 5, 4, 3, 2, 4, 3, 1, 2, 4, 3, 2, 4, 2, 1, 8, 7, 8, 6, 8, 7, 8, 7, 6, 5, 4, 3, 4, 3, 1, 2, 4, 2, 1, 8, 1, 3, 1, 8, 6, 7, 6, 5}
			} else if ss.runnum == 16 {
				seqid = [900]int{8, 6, 7, 5, 4, 2, 3, 2, 3, 1, 2, 3, 1, 2, 4, 5, 4, 3, 4, 3, 1, 3, 1, 8, 7, 8, 6, 7, 6, 5, 7, 8, 6, 7, 6, 7, 6, 8, 7, 8, 1, 8, 7, 8, 7, 5, 7, 5, 7, 6, 7, 6, 5, 4, 3, 1, 8, 6, 8, 6, 7, 6, 8, 7, 8, 1, 8, 7, 8, 7, 5, 6, 7, 5, 7, 6, 8, 6, 8, 1, 3, 4, 2, 1, 3, 1, 8, 1, 3, 1, 3, 1, 8, 1, 8, 7, 8, 6, 8, 1, 3, 1, 3, 2, 3, 4, 5, 6, 5, 4, 3, 4, 2, 1, 3, 1, 3, 1, 2, 4, 5, 6, 5, 7, 8, 7, 8, 6, 5, 6, 5, 6, 7, 5, 7, 6, 5, 7, 5, 6, 8, 7, 6, 7, 8, 7, 6, 8, 1, 8, 1, 8, 1, 2, 1, 8, 1, 3, 1, 3, 2, 1, 3, 1, 2, 3, 4, 3, 1, 3, 2, 1, 8, 1, 3, 2, 1, 3, 1, 3, 2, 4, 3, 4, 3, 1, 3, 4, 2, 3, 1, 8, 7, 5, 4, 2, 1, 2, 1, 3, 4, 5, 7, 6, 7, 8, 7, 6, 8, 7, 5, 7, 6, 5, 7, 5, 7, 5, 4, 2, 3, 4, 5, 6, 8, 6, 8, 1, 3, 1, 8, 7, 8, 1, 3, 2, 4, 3, 2, 3, 2, 4, 3, 1, 3, 4, 3, 2, 4, 3, 4, 3, 4, 3, 2, 1, 2, 1, 8, 6, 5, 6, 8, 6, 8, 6, 7, 5, 6, 8, 7, 5, 6, 5, 6, 5, 4, 5, 4, 2, 4, 3, 1, 3, 2, 1, 8, 7, 8, 7, 6, 5, 4, 2, 4, 3, 2, 4, 2, 4, 2, 3, 2, 1, 3, 2, 4, 2, 1, 2, 4, 2, 3, 1, 3, 1, 2, 3, 1, 8, 7, 8, 7, 5, 6, 5, 7, 6, 8, 1, 8, 1, 2, 3, 4, 3, 1, 8, 7, 8, 7, 5, 4, 3, 2, 3, 4, 3, 4, 3, 4, 3, 2, 3, 4, 5, 4, 5, 7, 8, 7, 8, 7, 8, 6, 7, 5, 7, 6, 5, 7, 8, 7, 8, 1, 8, 6, 5, 6, 5, 6, 5, 4, 5, 4, 3, 2, 3, 2, 3, 1, 8, 7, 6, 5, 4, 5, 7, 5, 6, 8, 6, 8, 7, 5, 7, 6, 7, 5, 4, 2, 4, 2, 1, 2, 3, 2, 1, 8, 1, 3, 1, 2, 3, 4, 5, 7, 5, 7, 5, 7, 8, 6, 7, 8, 7, 6, 7, 5, 7, 5, 7, 6, 5, 4, 3, 2, 1, 8, 1, 3, 2, 1, 3, 1, 8, 1, 8, 1, 3, 4, 3, 1, 3, 2, 1, 3, 4, 3, 1, 3, 4, 5, 6, 5, 6, 8, 6, 5, 4, 3, 4, 2, 4, 3, 4, 2, 1, 2, 4, 3, 4, 3, 4, 3, 2, 1, 8, 1, 8, 1, 3, 1, 8, 1, 3, 4, 2, 4, 5, 6, 7, 8, 7, 5, 6, 8, 1, 3, 4, 5, 4, 5, 4, 3, 1, 3, 1, 3, 1, 8, 6, 7, 8, 7, 6, 8, 7, 6, 8, 1, 3, 4, 2, 1, 2, 4, 2, 1, 8, 1, 8, 7, 5, 6, 8, 1, 8, 1, 3, 1, 2, 4, 5, 4, 3, 4, 3, 1, 2, 1, 3, 2, 4, 5, 4, 3, 2, 1, 8, 1, 3, 2, 1, 8, 6, 8, 7, 6, 8, 7, 5, 6, 7, 5, 4, 2, 3, 2, 3, 4, 3, 2, 4, 5, 4, 2, 1, 8, 7, 6, 5, 7, 6, 8, 6, 8, 6, 8, 1, 8, 7, 6, 8, 6, 7, 5, 6, 8, 6, 5, 4, 2, 1, 3, 4, 3, 1, 8, 6, 5, 7, 5, 4, 2, 1, 3, 2, 1, 8, 6, 5, 7, 6, 7, 5, 6, 7, 5, 7, 6, 5, 4, 3, 2, 3, 4, 5, 7, 8, 7, 5, 7, 6, 7, 6, 8, 1, 8, 7, 6, 5, 4, 5, 4, 3, 4, 3, 4, 5, 4, 5, 6, 8, 1, 3, 1, 8, 1, 3, 4, 5, 6, 7, 8, 7, 5, 7, 5, 6, 5, 6, 8, 7, 6, 5, 7, 6, 5, 7, 8, 1, 3, 2, 3, 4, 3, 4, 2, 4, 2, 4, 2, 3, 2, 4, 5, 7, 5, 7, 8, 7, 6, 7, 8, 1, 2, 3, 2, 3, 4, 3, 4, 5, 7, 6, 8, 6, 5, 6, 5, 4, 5, 6, 5, 4, 3, 1, 2, 1, 3, 4, 2, 3, 1, 2, 4, 2, 3, 2, 4, 3, 1, 8, 7, 6, 8, 7, 6, 5, 4, 3, 2, 4, 2, 1, 3, 2, 3, 2, 3, 4, 3, 1, 2, 1, 8, 6, 8, 1, 2, 3, 4, 5, 4, 3, 4, 3, 4, 2, 4, 2, 3, 1, 2, 3, 4, 3, 2, 1, 2, 4, 3, 4, 5, 6, 5, 6, 5, 7, 5, 7, 6, 5, 6, 5, 6, 7, 5, 7, 5, 4, 2, 4, 2, 3, 4, 3, 2, 4, 3, 4, 5, 4, 5, 6, 5, 6, 5, 7, 6, 5, 6, 8, 1, 8, 6, 7, 6, 8, 7, 5, 4, 5, 7, 6, 7, 5, 6, 8, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1}
			} else if ss.runnum == 17 {
				seqid = [900]int{3, 4, 2, 1, 8, 1, 2, 3, 2, 3, 4, 5, 4, 3, 1, 8, 6, 5, 7, 5, 6, 7, 6, 5, 6, 5, 4, 5, 6, 8, 1, 8, 7, 5, 6, 8, 1, 3, 1, 2, 1, 2, 3, 1, 2, 4, 3, 1, 8, 7, 8, 1, 2, 4, 2, 4, 5, 6, 8, 6, 8, 1, 8, 7, 6, 7, 8, 6, 7, 6, 8, 1, 8, 1, 8, 7, 8, 6, 8, 1, 8, 7, 8, 6, 8, 6, 7, 8, 7, 5, 7, 6, 5, 4, 2, 4, 3, 4, 3, 2, 3, 2, 1, 3, 1, 3, 4, 3, 1, 8, 6, 8, 6, 7, 6, 5, 4, 3, 2, 3, 1, 8, 1, 8, 1, 3, 1, 2, 1, 3, 2, 1, 2, 1, 2, 4, 5, 7, 8, 6, 7, 8, 1, 2, 3, 4, 5, 4, 2, 3, 4, 5, 7, 8, 7, 6, 5, 4, 2, 4, 3, 2, 1, 8, 7, 5, 6, 8, 7, 8, 7, 6, 7, 5, 7, 5, 4, 5, 4, 5, 4, 2, 1, 2, 3, 1, 2, 1, 8, 6, 8, 7, 5, 6, 8, 6, 8, 1, 2, 4, 5, 4, 3, 4, 5, 6, 8, 6, 5, 7, 5, 7, 8, 1, 8, 7, 5, 7, 5, 4, 5, 7, 5, 6, 5, 6, 5, 7, 8, 6, 7, 6, 5, 7, 6, 5, 6, 7, 8, 1, 2, 4, 3, 1, 3, 4, 5, 4, 2, 4, 5, 7, 6, 7, 5, 4, 2, 1, 8, 6, 5, 4, 3, 1, 3, 4, 3, 2, 4, 5, 4, 5, 4, 2, 4, 3, 1, 2, 4, 5, 6, 8, 6, 8, 1, 8, 7, 8, 6, 7, 6, 7, 5, 4, 5, 6, 7, 8, 7, 8, 7, 6, 8, 6, 7, 5, 7, 5, 6, 7, 8, 6, 7, 6, 8, 1, 8, 7, 8, 6, 8, 1, 2, 3, 1, 2, 3, 4, 2, 1, 2, 3, 4, 5, 4, 5, 7, 8, 6, 5, 7, 5, 4, 5, 4, 3, 2, 4, 5, 7, 8, 7, 6, 8, 1, 8, 6, 7, 6, 7, 6, 5, 7, 8, 7, 8, 1, 8, 6, 5, 7, 8, 7, 8, 1, 3, 4, 3, 4, 3, 4, 2, 4, 2, 3, 1, 3, 1, 3, 4, 3, 2, 3, 2, 4, 2, 4, 3, 4, 5, 4, 3, 1, 3, 4, 5, 7, 6, 8, 6, 8, 7, 6, 5, 7, 5, 6, 7, 5, 7, 5, 6, 8, 7, 6, 8, 6, 5, 4, 3, 1, 8, 6, 5, 7, 5, 4, 5, 6, 8, 1, 2, 4, 3, 1, 8, 1, 3, 1, 3, 1, 8, 7, 8, 1, 3, 4, 3, 1, 8, 1, 2, 1, 2, 1, 8, 7, 6, 7, 5, 7, 6, 8, 7, 5, 6, 7, 8, 1, 8, 1, 3, 2, 4, 3, 4, 2, 4, 2, 1, 3, 4, 3, 1, 8, 6, 8, 6, 7, 5, 7, 6, 5, 6, 8, 7, 6, 7, 6, 7, 6, 5, 4, 5, 7, 5, 4, 2, 1, 2, 3, 4, 5, 7, 6, 8, 6, 7, 5, 4, 5, 7, 5, 7, 6, 5, 7, 5, 6, 7, 8, 1, 3, 4, 2, 1, 8, 6, 5, 4, 3, 4, 2, 1, 3, 4, 2, 3, 2, 1, 2, 3, 2, 1, 8, 6, 7, 6, 8, 1, 2, 3, 1, 2, 4, 2, 4, 5, 7, 5, 4, 2, 3, 1, 2, 4, 2, 1, 3, 4, 3, 2, 1, 8, 7, 6, 5, 7, 8, 6, 8, 7, 6, 7, 8, 7, 8, 7, 5, 7, 8, 6, 7, 5, 6, 7, 6, 8, 1, 8, 6, 7, 5, 6, 5, 7, 8, 7, 8, 7, 5, 7, 5, 4, 5, 7, 5, 6, 8, 7, 6, 5, 6, 5, 4, 3, 1, 2, 3, 2, 4, 3, 4, 5, 4, 3, 4, 3, 1, 2, 1, 8, 6, 7, 5, 4, 2, 3, 1, 2, 4, 3, 4, 2, 3, 2, 3, 1, 8, 1, 8, 1, 8, 1, 2, 1, 3, 4, 5, 7, 5, 7, 5, 7, 5, 4, 5, 6, 8, 6, 8, 7, 5, 6, 8, 7, 6, 5, 7, 5, 6, 8, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 3, 4, 5, 6, 5, 7, 6, 8, 1, 2, 3, 1, 8, 7, 6, 8, 6, 7, 6, 5, 6, 5, 4, 2, 4, 2, 1, 8, 6, 7, 6, 7, 6, 8, 1, 8, 6, 5, 6, 5, 6, 5, 6, 7, 5, 7, 8, 7, 8, 1, 2, 3, 1, 8, 7, 8, 1, 2, 4, 5, 6, 7, 6, 5, 7, 6, 8, 7, 6, 8, 6, 7, 8, 6, 7, 5, 6, 7, 5, 4, 3, 1, 8, 6, 7, 8, 1, 8, 1, 3, 4, 3, 2, 1, 2, 4, 2, 1, 3, 4, 5, 7, 6, 8, 1, 2, 4, 2, 1, 2, 3, 2, 4, 5, 6, 5, 6, 8, 7, 8, 7, 5, 7, 6, 8, 7, 6, 7, 5, 7, 6, 7, 8, 7, 5, 6, 8, 1, 2, 1, 2, 4, 5, 7, 5, 7, 6, 7, 8, 7, 6, 7, 5, 7, 5, 7, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 6, 5, 6, 7, 5, 7, 5, 4, 5, 6, 5, 7}
			} else if ss.runnum == 18 {
				seqid = [900]int{8, 6, 7, 5, 7, 5, 7, 8, 1, 3, 4, 5, 4, 2, 1, 2, 3, 1, 3, 4, 5, 4, 2, 4, 2, 3, 1, 3, 4, 3, 1, 3, 2, 4, 2, 1, 2, 1, 3, 1, 8, 7, 8, 7, 8, 6, 7, 6, 8, 7, 5, 6, 8, 1, 8, 6, 7, 8, 7, 5, 7, 6, 7, 8, 7, 6, 7, 5, 4, 3, 1, 8, 7, 8, 1, 3, 2, 4, 5, 4, 3, 4, 3, 4, 3, 2, 4, 3, 4, 2, 4, 2, 3, 4, 5, 4, 5, 6, 8, 6, 5, 4, 3, 4, 3, 2, 3, 2, 3, 2, 3, 1, 8, 6, 8, 7, 8, 7, 8, 6, 8, 1, 2, 3, 2, 4, 3, 1, 8, 6, 8, 6, 8, 6, 5, 7, 8, 7, 8, 7, 6, 8, 7, 5, 4, 3, 1, 3, 2, 1, 8, 1, 8, 1, 8, 7, 6, 5, 4, 5, 7, 6, 5, 6, 7, 5, 7, 5, 6, 8, 1, 3, 1, 3, 1, 8, 1, 8, 7, 8, 1, 8, 7, 8, 1, 3, 2, 3, 4, 3, 4, 3, 2, 4, 5, 6, 7, 5, 4, 3, 2, 3, 2, 3, 2, 1, 8, 1, 8, 1, 3, 4, 3, 1, 3, 4, 2, 1, 3, 1, 2, 4, 5, 4, 2, 1, 3, 2, 1, 8, 7, 8, 1, 8, 6, 8, 7, 5, 4, 5, 6, 7, 5, 7, 6, 7, 8, 1, 2, 4, 2, 3, 2, 1, 3, 4, 3, 1, 3, 2, 1, 8, 6, 7, 8, 1, 3, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 2, 1, 2, 4, 5, 7, 5, 7, 5, 6, 5, 6, 8, 1, 3, 4, 3, 4, 2, 3, 4, 5, 4, 2, 3, 4, 2, 4, 3, 1, 3, 1, 8, 7, 5, 6, 8, 1, 8, 6, 8, 6, 7, 6, 5, 4, 5, 7, 6, 8, 1, 3, 2, 4, 3, 4, 3, 1, 8, 7, 5, 4, 5, 4, 5, 4, 3, 4, 5, 4, 2, 1, 3, 2, 4, 2, 1, 3, 1, 8, 7, 5, 4, 2, 1, 2, 4, 3, 2, 3, 4, 5, 6, 5, 6, 8, 7, 6, 8, 6, 8, 6, 5, 4, 3, 1, 8, 7, 6, 7, 5, 4, 3, 1, 8, 7, 6, 5, 4, 3, 1, 2, 3, 2, 3, 4, 5, 4, 2, 1, 2, 4, 3, 1, 8, 6, 5, 4, 3, 1, 3, 2, 3, 2, 1, 8, 7, 6, 8, 6, 7, 6, 8, 7, 5, 7, 5, 4, 5, 7, 8, 1, 2, 4, 2, 4, 2, 4, 2, 1, 3, 4, 5, 7, 8, 7, 5, 7, 8, 6, 5, 7, 8, 7, 6, 8, 7, 5, 6, 8, 6, 7, 8, 1, 3, 1, 2, 3, 2, 1, 2, 1, 2, 4, 3, 1, 8, 1, 3, 2, 3, 2, 3, 4, 2, 4, 2, 1, 3, 2, 1, 8, 7, 6, 7, 5, 6, 7, 6, 8, 7, 8, 6, 7, 5, 7, 5, 7, 6, 7, 8, 7, 6, 5, 4, 2, 1, 3, 1, 8, 6, 5, 6, 5, 4, 2, 4, 2, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 4, 5, 7, 8, 7, 8, 6, 7, 6, 8, 6, 5, 7, 6, 7, 6, 8, 7, 6, 7, 5, 4, 5, 6, 7, 8, 6, 5, 7, 6, 7, 6, 8, 7, 5, 4, 5, 7, 5, 6, 8, 6, 5, 6, 7, 5, 4, 5, 4, 5, 4, 5, 6, 5, 6, 8, 6, 8, 7, 8, 1, 8, 7, 6, 7, 6, 5, 4, 2, 1, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 2, 3, 4, 5, 4, 5, 7, 5, 4, 2, 1, 8, 6, 5, 7, 5, 4, 3, 1, 8, 6, 5, 7, 6, 8, 7, 8, 6, 5, 7, 8, 6, 5, 7, 5, 7, 5, 7, 6, 5, 4, 3, 2, 1, 8, 1, 2, 3, 1, 3, 4, 5, 4, 3, 1, 8, 1, 2, 4, 2, 3, 4, 5, 6, 5, 6, 5, 7, 8, 7, 5, 4, 2, 3, 1, 2, 4, 5, 6, 5, 6, 8, 6, 7, 6, 7, 8, 1, 8, 1, 8, 6, 8, 6, 5, 4, 5, 4, 2, 1, 8, 6, 8, 6, 8, 1, 8, 7, 6, 7, 5, 7, 8, 1, 8, 7, 6, 7, 5, 6, 8, 6, 8, 1, 8, 1, 8, 7, 6, 8, 1, 8, 6, 8, 7, 6, 8, 7, 8, 6, 8, 6, 7, 5, 7, 6, 8, 7, 6, 7, 8, 6, 7, 8, 6, 5, 7, 5, 4, 3, 4, 3, 2, 1, 3, 4, 5, 6, 5, 6, 7, 8, 1, 8, 1, 8, 7, 8, 6, 5, 7, 5, 4, 5, 4, 5, 4, 5, 7, 6, 8, 1, 3, 1, 3, 4, 5, 6, 7, 8, 6, 8, 7, 8, 7, 8, 7, 6, 8, 1, 8, 6, 7, 6, 7, 5, 6, 8, 6, 8, 6, 8, 7, 6, 8, 6, 8, 1, 2, 1, 2, 1, 3, 2, 4, 5, 7, 6, 8, 7, 5, 6, 8, 6, 5, 6, 7, 6, 7, 5, 4, 2, 1, 3, 4, 5, 6, 7, 5, 6, 7, 5, 6, 5, 7, 6, 8, 7, 6, 5, 6, 8, 6, 8}
			} else if ss.runnum == 19 {
				seqid = [900]int{2, 1, 2, 1, 3, 4, 2, 1, 8, 7, 5, 7, 5, 4, 3, 2, 1, 2, 3, 4, 5, 7, 5, 4, 5, 6, 5, 6, 7, 8, 6, 7, 5, 6, 8, 1, 8, 7, 8, 6, 8, 1, 8, 6, 7, 5, 7, 5, 4, 3, 4, 5, 6, 7, 5, 4, 2, 3, 4, 3, 2, 4, 5, 6, 8, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 1, 3, 2, 4, 2, 1, 3, 1, 8, 6, 5, 4, 5, 7, 8, 6, 5, 6, 8, 1, 8, 6, 7, 5, 7, 5, 7, 5, 7, 8, 6, 7, 6, 5, 6, 5, 6, 8, 6, 7, 8, 7, 6, 5, 7, 8, 7, 8, 7, 8, 1, 3, 4, 3, 2, 4, 3, 2, 4, 3, 2, 1, 8, 1, 3, 1, 3, 1, 2, 1, 2, 4, 5, 7, 8, 6, 8, 6, 7, 5, 6, 8, 1, 3, 4, 5, 7, 8, 1, 8, 6, 8, 1, 3, 1, 2, 4, 2, 4, 2, 3, 4, 2, 4, 5, 6, 8, 6, 7, 5, 4, 3, 4, 2, 4, 2, 1, 8, 7, 6, 8, 6, 8, 6, 8, 7, 5, 7, 8, 1, 3, 1, 2, 3, 1, 3, 2, 4, 3, 4, 3, 4, 5, 4, 3, 2, 4, 3, 1, 8, 7, 8, 7, 8, 7, 8, 1, 8, 6, 5, 4, 3, 4, 2, 4, 5, 4, 3, 1, 3, 4, 3, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 4, 2, 1, 8, 7, 6, 8, 1, 2, 4, 3, 2, 3, 4, 5, 6, 5, 7, 8, 6, 5, 4, 5, 7, 8, 1, 3, 1, 3, 2, 3, 4, 2, 4, 5, 7, 8, 1, 8, 1, 8, 1, 8, 7, 8, 1, 3, 2, 3, 2, 1, 2, 1, 8, 6, 8, 6, 8, 7, 8, 1, 2, 4, 3, 4, 5, 4, 3, 4, 3, 2, 4, 3, 2, 4, 2, 1, 2, 1, 3, 4, 5, 4, 5, 4, 2, 3, 4, 3, 1, 2, 1, 3, 4, 2, 3, 1, 3, 1, 3, 2, 1, 8, 6, 7, 6, 8, 6, 7, 8, 6, 7, 6, 5, 7, 8, 1, 2, 3, 4, 3, 1, 8, 7, 8, 6, 8, 7, 5, 4, 5, 4, 2, 1, 8, 7, 5, 4, 3, 4, 5, 4, 5, 7, 5, 6, 5, 7, 8, 7, 6, 5, 7, 6, 7, 5, 7, 6, 8, 7, 5, 7, 6, 8, 7, 5, 4, 2, 1, 8, 1, 3, 2, 4, 2, 4, 5, 7, 8, 7, 5, 6, 5, 6, 8, 6, 8, 1, 3, 1, 8, 1, 2, 1, 2, 1, 3, 1, 2, 3, 2, 1, 8, 7, 8, 7, 5, 7, 6, 8, 7, 6, 8, 6, 5, 7, 8, 6, 5, 4, 3, 4, 2, 1, 8, 1, 8, 1, 2, 1, 2, 4, 2, 1, 3, 4, 3, 1, 3, 4, 5, 4, 2, 4, 2, 4, 3, 1, 3, 2, 4, 3, 1, 3, 4, 5, 4, 5, 6, 5, 6, 8, 1, 3, 4, 2, 4, 5, 6, 8, 1, 3, 1, 8, 1, 2, 1, 3, 4, 3, 1, 3, 2, 4, 5, 6, 7, 8, 6, 5, 4, 2, 3, 1, 3, 4, 2, 4, 2, 4, 3, 1, 8, 6, 8, 6, 5, 4, 2, 4, 3, 1, 2, 3, 4, 5, 7, 8, 6, 5, 7, 8, 6, 7, 8, 6, 8, 7, 5, 7, 6, 8, 7, 6, 5, 6, 5, 6, 8, 6, 5, 4, 2, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 1, 8, 6, 5, 6, 7, 6, 7, 5, 4, 5, 7, 6, 8, 6, 8, 7, 6, 8, 1, 3, 1, 8, 7, 6, 8, 6, 7, 5, 6, 7, 6, 5, 7, 8, 6, 8, 1, 8, 1, 8, 6, 5, 4, 5, 6, 7, 8, 6, 8, 7, 8, 1, 8, 6, 7, 6, 7, 8, 1, 3, 1, 2, 1, 2, 3, 2, 3, 4, 3, 2, 4, 2, 4, 5, 6, 5, 7, 6, 7, 5, 6, 8, 1, 8, 7, 6, 8, 1, 3, 2, 4, 5, 6, 8, 6, 7, 5, 7, 6, 8, 7, 6, 7, 5, 4, 2, 3, 2, 1, 8, 1, 3, 4, 5, 4, 2, 4, 5, 4, 2, 1, 2, 1, 2, 4, 2, 4, 3, 1, 3, 4, 3, 4, 5, 6, 8, 7, 5, 7, 5, 4, 3, 4, 2, 4, 2, 3, 4, 2, 4, 3, 4, 2, 4, 5, 4, 5, 6, 7, 6, 5, 7, 8, 6, 7, 5, 7, 5, 6, 7, 8, 1, 2, 4, 3, 4, 3, 4, 3, 2, 1, 8, 7, 6, 5, 6, 8, 6, 7, 8, 6, 5, 6, 8, 1, 2, 1, 8, 7, 8, 1, 3, 1, 2, 3, 1, 3, 1, 8, 1, 3, 2, 1, 3, 4, 3, 2, 3, 4, 2, 1, 2, 1, 8, 7, 8, 7, 6, 8, 1, 2, 1, 8, 6, 5, 7, 5, 6, 7, 5, 7, 6, 5, 6, 8, 1, 8, 6, 8, 6, 7, 5, 4, 5, 6, 7, 5, 7, 8, 6, 7, 5, 6, 5, 7, 8, 1, 3, 2, 4, 2, 3, 1, 3, 1, 3, 2, 3, 2, 3, 2, 3, 4, 3, 1, 8}
			} else if ss.runnum == 20 {
				seqid = [900]int{2, 3, 4, 2, 4, 3, 2, 4, 3, 1, 2, 4, 5, 4, 2, 4, 5, 6, 8, 1, 8, 1, 3, 1, 8, 6, 8, 6, 5, 4, 3, 1, 2, 4, 5, 6, 5, 4, 2, 3, 4, 2, 1, 2, 3, 1, 2, 3, 4, 2, 1, 8, 1, 2, 1, 2, 3, 4, 2, 3, 2, 4, 3, 4, 3, 2, 1, 8, 1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 7, 8, 7, 5, 7, 8, 6, 8, 1, 2, 1, 2, 1, 3, 4, 2, 4, 2, 1, 3, 4, 2, 3, 4, 3, 2, 1, 3, 2, 4, 3, 1, 3, 4, 3, 1, 2, 1, 2, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6, 8, 7, 6, 8, 6, 7, 8, 6, 7, 8, 6, 7, 5, 7, 8, 6, 7, 5, 7, 6, 5, 4, 3, 2, 3, 4, 3, 2, 3, 4, 2, 3, 2, 3, 4, 3, 2, 3, 2, 4, 5, 4, 2, 3, 4, 3, 2, 4, 2, 4, 5, 7, 5, 6, 5, 7, 5, 4, 3, 1, 2, 1, 2, 4, 3, 2, 4, 3, 2, 1, 3, 1, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 4, 2, 1, 3, 2, 1, 8, 7, 6, 5, 6, 5, 6, 8, 1, 3, 1, 8, 6, 8, 7, 6, 7, 6, 8, 7, 8, 1, 8, 7, 8, 1, 3, 2, 1, 8, 1, 3, 1, 3, 1, 3, 1, 2, 3, 1, 8, 1, 3, 2, 4, 2, 4, 5, 4, 5, 4, 2, 3, 4, 2, 1, 8, 1, 2, 3, 1, 3, 1, 2, 1, 8, 1, 3, 2, 1, 8, 1, 3, 2, 4, 5, 7, 6, 7, 8, 6, 8, 6, 7, 6, 5, 6, 7, 8, 7, 6, 8, 1, 8, 1, 3, 4, 2, 1, 8, 6, 8, 6, 7, 6, 8, 6, 7, 8, 1, 3, 1, 8, 1, 3, 4, 5, 7, 5, 7, 8, 6, 8, 7, 6, 8, 7, 6, 8, 1, 2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 4, 5, 6, 8, 6, 5, 6, 7, 6, 5, 6, 8, 7, 8, 1, 8, 1, 3, 4, 2, 3, 1, 3, 4, 2, 3, 4, 3, 4, 3, 1, 2, 3, 1, 8, 7, 5, 6, 7, 8, 7, 6, 8, 1, 8, 1, 2, 1, 3, 1, 8, 7, 8, 6, 8, 1, 8, 6, 8, 7, 8, 1, 3, 1, 3, 1, 3, 2, 1, 8, 6, 5, 7, 6, 7, 6, 5, 4, 5, 7, 6, 8, 7, 8, 1, 3, 1, 3, 4, 2, 4, 2, 4, 3, 1, 3, 2, 4, 3, 2, 4, 2, 4, 5, 7, 8, 1, 2, 3, 1, 8, 6, 5, 7, 5, 7, 8, 6, 8, 7, 5, 6, 5, 6, 7, 5, 4, 5, 7, 5, 6, 8, 6, 7, 5, 4, 3, 2, 4, 2, 3, 2, 1, 8, 6, 7, 5, 4, 5, 6, 5, 7, 5, 6, 8, 6, 8, 1, 3, 1, 3, 2, 4, 5, 6, 7, 6, 7, 5, 6, 7, 6, 7, 6, 7, 5, 6, 8, 6, 5, 4, 2, 3, 1, 8, 7, 8, 6, 7, 6, 7, 5, 4, 3, 2, 1, 2, 1, 3, 2, 3, 2, 4, 2, 3, 2, 1, 8, 6, 5, 4, 2, 4, 2, 1, 8, 7, 6, 5, 6, 5, 6, 7, 5, 7, 6, 7, 6, 7, 5, 7, 6, 5, 4, 3, 4, 3, 4, 5, 4, 3, 4, 2, 1, 3, 2, 3, 1, 3, 4, 2, 3, 2, 4, 3, 4, 3, 1, 3, 2, 4, 2, 3, 2, 3, 4, 3, 2, 3, 1, 2, 4, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 8, 6, 8, 7, 6, 7, 6, 8, 6, 7, 8, 1, 2, 4, 3, 4, 5, 7, 8, 6, 5, 4, 3, 4, 2, 3, 1, 2, 3, 2, 1, 8, 1, 3, 4, 3, 4, 2, 3, 2, 1, 3, 1, 8, 6, 5, 6, 8, 6, 5, 4, 3, 1, 8, 7, 6, 7, 8, 7, 6, 5, 4, 2, 1, 8, 1, 2, 1, 2, 4, 3, 2, 1, 8, 6, 7, 6, 5, 6, 7, 5, 7, 5, 4, 5, 4, 2, 4, 3, 4, 3, 2, 1, 8, 6, 5, 6, 8, 7, 6, 8, 1, 2, 4, 2, 4, 3, 4, 3, 1, 3, 4, 5, 6, 7, 6, 8, 7, 6, 7, 8, 6, 7, 5, 4, 2, 1, 2, 1, 2, 1, 2, 3, 2, 4, 5, 6, 7, 5, 7, 6, 8, 6, 8, 6, 7, 6, 7, 6, 8, 6, 5, 4, 3, 1, 3, 4, 5, 4, 3, 2, 3, 2, 1, 3, 4, 3, 1, 2, 1, 2, 1, 3, 2, 4, 5, 7, 8, 6, 7, 8, 6, 8, 7, 5, 6, 5, 4, 5, 4, 3, 2, 1, 3, 2, 1, 8, 6, 8, 7, 8, 1, 2, 4, 5, 6, 8, 6, 5, 7, 5, 6, 5, 7, 6, 7, 6, 7, 8, 7, 6, 7, 6, 8, 1, 8, 7, 8, 6, 7, 8, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 1, 2, 1, 3, 1, 2, 3, 2, 3, 4, 5, 7, 6, 7, 8, 7, 5, 7, 5, 6}
			} else if ss.runnum == 21 {
				seqid = [900]int{1, 8, 6, 7, 6, 5, 6, 8, 6, 5, 4, 3, 1, 3, 2, 4, 3, 2, 3, 1, 3, 4, 5, 7, 5, 7, 5, 4, 2, 1, 2, 3, 1, 8, 1, 2, 3, 2, 1, 2, 4, 2, 1, 3, 4, 3, 4, 5, 6, 5, 7, 5, 6, 7, 5, 6, 5, 6, 7, 6, 5, 4, 3, 1, 3, 4, 5, 4, 3, 4, 3, 1, 3, 4, 5, 7, 8, 7, 5, 4, 2, 1, 2, 1, 3, 4, 2, 1, 3, 1, 2, 3, 4, 3, 1, 8, 6, 5, 4, 2, 4, 3, 1, 2, 4, 5, 7, 8, 6, 7, 6, 5, 7, 8, 7, 5, 6, 5, 6, 8, 7, 6, 5, 4, 5, 6, 8, 6, 7, 5, 4, 3, 1, 3, 1, 2, 1, 8, 7, 8, 7, 5, 7, 6, 7, 5, 7, 5, 7, 8, 1, 3, 1, 8, 7, 8, 1, 2, 3, 1, 8, 1, 8, 7, 8, 6, 8, 6, 5, 7, 6, 7, 8, 7, 6, 7, 6, 7, 5, 4, 5, 6, 7, 6, 7, 5, 6, 7, 5, 7, 8, 1, 8, 1, 2, 1, 3, 1, 3, 4, 5, 4, 3, 1, 3, 2, 3, 4, 3, 2, 3, 2, 1, 8, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 3, 1, 3, 4, 3, 1, 2, 4, 3, 4, 2, 3, 1, 8, 1, 2, 4, 2, 4, 2, 1, 3, 4, 2, 1, 8, 6, 5, 4, 2, 3, 4, 3, 4, 5, 6, 5, 7, 6, 8, 7, 8, 1, 2, 1, 2, 3, 4, 2, 1, 2, 3, 4, 3, 1, 3, 4, 5, 6, 7, 6, 8, 7, 8, 7, 8, 1, 2, 1, 8, 1, 3, 2, 3, 2, 3, 2, 4, 5, 6, 8, 6, 5, 7, 6, 7, 8, 7, 6, 5, 7, 6, 7, 5, 7, 5, 6, 7, 8, 1, 8, 6, 7, 8, 7, 5, 4, 3, 1, 2, 1, 8, 7, 5, 6, 5, 6, 8, 6, 5, 7, 5, 4, 2, 3, 4, 3, 1, 8, 1, 3, 4, 3, 1, 2, 4, 2, 3, 2, 3, 4, 2, 4, 2, 1, 3, 1, 8, 7, 5, 7, 8, 1, 3, 4, 3, 2, 1, 8, 1, 8, 1, 2, 1, 2, 1, 8, 6, 7, 8, 1, 3, 1, 3, 2, 1, 2, 3, 4, 5, 7, 6, 8, 6, 8, 7, 5, 6, 8, 1, 2, 3, 1, 8, 1, 8, 1, 2, 3, 2, 3, 1, 8, 1, 3, 1, 3, 4, 3, 1, 2, 1, 2, 3, 1, 2, 1, 8, 1, 2, 3, 4, 5, 7, 8, 1, 2, 1, 3, 4, 5, 6, 8, 6, 8, 7, 6, 5, 6, 8, 7, 8, 6, 7, 5, 7, 6, 5, 7, 5, 4, 3, 4, 2, 1, 8, 1, 8, 6, 7, 6, 5, 4, 2, 4, 5, 4, 3, 4, 3, 2, 4, 3, 1, 3, 1, 8, 1, 2, 4, 5, 7, 6, 5, 6, 5, 4, 5, 6, 8, 1, 8, 1, 8, 6, 5, 6, 7, 5, 6, 8, 6, 5, 7, 5, 6, 8, 1, 2, 3, 2, 4, 3, 1, 3, 2, 1, 2, 3, 1, 2, 4, 5, 4, 3, 2, 4, 2, 1, 3, 1, 2, 4, 3, 1, 8, 6, 8, 6, 5, 4, 2, 3, 1, 2, 3, 4, 2, 3, 2, 3, 1, 2, 4, 3, 1, 3, 4, 5, 4, 2, 1, 2, 1, 8, 1, 2, 4, 3, 1, 3, 2, 3, 1, 3, 4, 2, 4, 5, 4, 5, 7, 5, 4, 3, 2, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 4, 5, 7, 6, 7, 8, 6, 8, 7, 5, 4, 5, 6, 8, 7, 6, 7, 8, 1, 8, 7, 5, 6, 5, 7, 6, 7, 8, 6, 8, 6, 8, 7, 6, 5, 6, 7, 5, 7, 6, 8, 1, 2, 1, 8, 7, 6, 8, 1, 3, 2, 1, 8, 7, 5, 7, 8, 1, 3, 2, 3, 1, 2, 3, 4, 3, 1, 3, 4, 3, 4, 2, 1, 3, 1, 3, 1, 2, 3, 2, 1, 8, 1, 8, 7, 6, 8, 6, 7, 8, 7, 5, 7, 8, 1, 3, 4, 3, 1, 3, 2, 1, 3, 4, 3, 2, 4, 3, 4, 3, 1, 2, 1, 8, 6, 8, 6, 7, 8, 6, 7, 6, 5, 6, 5, 7, 5, 7, 8, 6, 5, 7, 6, 7, 6, 8, 6, 8, 7, 5, 4, 5, 6, 8, 6, 7, 6, 7, 5, 6, 7, 8, 1, 8, 1, 8, 6, 8, 7, 8, 6, 8, 7, 8, 1, 8, 1, 3, 1, 2, 1, 2, 1, 2, 3, 2, 4, 3, 2, 4, 3, 2, 3, 2, 4, 2, 1, 2, 4, 2, 3, 4, 2, 3, 1, 3, 2, 3, 2, 3, 1, 8, 6, 8, 7, 5, 7, 5, 6, 5, 6, 7, 6, 8, 6, 8, 6, 7, 5, 6, 8, 6, 8, 7, 5, 6, 5, 6, 7, 6, 5, 7, 5, 7, 8, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 5, 7, 8, 7, 5, 4, 2, 4, 3, 1, 3, 1, 2, 4, 2, 4, 2, 3, 2, 3, 4, 3, 1, 3, 4}
			} else if ss.runnum == 22 {
				seqid = [900]int{4, 2, 4, 5, 4, 5, 6, 7, 8, 1, 8, 6, 7, 6, 8, 1, 8, 7, 5, 7, 6, 7, 6, 5, 7, 8, 6, 7, 6, 5, 6, 5, 7, 8, 6, 5, 4, 5, 4, 3, 4, 2, 3, 4, 5, 4, 3, 2, 3, 1, 3, 4, 2, 3, 1, 2, 1, 3, 4, 5, 4, 2, 1, 3, 2, 4, 5, 4, 2, 3, 1, 3, 4, 5, 6, 8, 6, 8, 6, 5, 6, 5, 4, 2, 3, 2, 4, 5, 4, 2, 3, 2, 1, 8, 6, 7, 5, 6, 7, 8, 1, 2, 1, 3, 4, 5, 6, 5, 6, 5, 4, 2, 1, 8, 1, 8, 7, 6, 8, 7, 6, 5, 7, 6, 7, 8, 6, 7, 6, 8, 1, 3, 1, 3, 2, 1, 3, 1, 8, 1, 3, 1, 2, 3, 2, 1, 3, 4, 3, 4, 3, 1, 2, 3, 1, 8, 7, 6, 8, 1, 8, 6, 5, 4, 2, 4, 3, 2, 3, 4, 2, 1, 2, 3, 1, 2, 4, 2, 4, 2, 1, 2, 3, 2, 4, 3, 1, 2, 1, 8, 1, 3, 4, 2, 4, 5, 4, 3, 2, 4, 3, 1, 8, 7, 6, 7, 8, 7, 8, 1, 8, 6, 7, 6, 7, 6, 8, 1, 8, 6, 5, 7, 6, 5, 4, 5, 6, 5, 6, 8, 6, 5, 7, 5, 6, 5, 4, 5, 6, 5, 7, 5, 6, 7, 5, 4, 2, 4, 2, 3, 1, 8, 6, 8, 7, 6, 8, 6, 5, 6, 8, 1, 8, 1, 8, 7, 5, 6, 7, 8, 7, 5, 6, 5, 7, 5, 6, 8, 7, 5, 7, 6, 8, 7, 8, 7, 6, 7, 6, 7, 8, 1, 8, 6, 8, 1, 8, 7, 5, 6, 8, 7, 5, 4, 2, 4, 3, 2, 4, 5, 6, 8, 6, 8, 7, 6, 5, 4, 5, 7, 8, 6, 5, 6, 7, 6, 8, 1, 2, 4, 5, 6, 5, 4, 3, 4, 2, 1, 2, 1, 8, 7, 6, 8, 1, 2, 1, 8, 1, 8, 6, 5, 6, 5, 4, 2, 1, 8, 6, 5, 6, 7, 8, 7, 6, 5, 4, 5, 6, 5, 7, 6, 5, 6, 7, 6, 8, 6, 8, 7, 5, 7, 5, 4, 5, 4, 5, 6, 7, 5, 7, 6, 5, 7, 5, 4, 3, 2, 4, 3, 2, 3, 4, 3, 1, 3, 4, 3, 4, 2, 1, 3, 4, 5, 7, 6, 5, 7, 8, 6, 8, 6, 5, 6, 8, 1, 2, 1, 2, 1, 8, 6, 5, 4, 5, 4, 3, 1, 8, 7, 5, 4, 2, 3, 4, 2, 1, 3, 2, 4, 5, 4, 3, 1, 2, 3, 2, 4, 5, 7, 8, 7, 8, 7, 5, 6, 7, 5, 7, 8, 7, 6, 7, 8, 7, 8, 7, 6, 7, 5, 4, 2, 4, 5, 6, 8, 6, 5, 6, 5, 7, 6, 7, 5, 4, 5, 4, 2, 4, 5, 6, 5, 7, 6, 7, 6, 5, 4, 3, 4, 2, 4, 3, 4, 5, 4, 3, 1, 8, 7, 8, 1, 3, 4, 5, 7, 6, 5, 4, 2, 1, 3, 2, 4, 2, 4, 3, 2, 1, 2, 3, 2, 1, 8, 1, 8, 7, 5, 7, 6, 7, 8, 7, 8, 7, 6, 7, 6, 5, 4, 3, 4, 5, 4, 3, 4, 2, 1, 3, 4, 5, 6, 7, 8, 6, 5, 7, 5, 6, 5, 7, 6, 7, 6, 8, 1, 2, 1, 8, 1, 2, 1, 2, 3, 1, 8, 6, 5, 6, 8, 1, 3, 1, 8, 6, 7, 8, 7, 6, 7, 5, 6, 5, 6, 8, 1, 2, 3, 1, 8, 6, 8, 6, 7, 5, 6, 8, 7, 8, 7, 8, 7, 6, 5, 4, 3, 1, 8, 7, 5, 4, 3, 1, 2, 1, 3, 2, 4, 3, 1, 8, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 2, 3, 2, 1, 8, 7, 5, 4, 5, 4, 2, 3, 4, 3, 4, 3, 4, 2, 1, 3, 2, 3, 2, 1, 3, 2, 1, 3, 2, 4, 3, 4, 5, 6, 7, 6, 5, 6, 7, 8, 1, 2, 4, 5, 6, 5, 4, 3, 1, 8, 6, 7, 5, 4, 2, 3, 4, 2, 4, 2, 1, 2, 4, 3, 2, 1, 8, 1, 3, 2, 4, 2, 4, 3, 1, 8, 7, 6, 7, 5, 7, 5, 7, 6, 7, 5, 6, 7, 6, 5, 4, 2, 3, 4, 5, 7, 8, 6, 7, 6, 5, 7, 5, 6, 8, 6, 7, 6, 7, 6, 7, 8, 6, 5, 7, 5, 7, 6, 7, 6, 7, 5, 6, 5, 6, 8, 7, 6, 5, 6, 8, 6, 8, 6, 7, 5, 6, 5, 4, 5, 7, 6, 5, 7, 6, 5, 7, 5, 6, 7, 6, 7, 6, 7, 6, 5, 4, 3, 1, 8, 7, 8, 6, 5, 7, 6, 8, 1, 3, 1, 3, 4, 5, 6, 5, 4, 2, 4, 5, 6, 8, 7, 8, 7, 6, 8, 1, 3, 2, 1, 2, 3, 4, 5, 4, 2, 4, 2, 3, 4, 5, 4, 2, 1, 2, 4, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 1, 8, 1, 2, 1, 3, 1, 8, 6, 5, 4, 5, 6, 8, 7, 8, 7, 8, 6, 8, 1, 8}
			} else if ss.runnum == 23 {
				seqid = [900]int{8, 6, 8, 1, 2, 1, 2, 3, 1, 3, 1, 2, 1, 2, 4, 5, 7, 6, 5, 6, 8, 1, 2, 3, 1, 8, 7, 8, 7, 5, 7, 5, 7, 6, 5, 6, 8, 1, 3, 4, 5, 4, 3, 4, 5, 4, 5, 4, 3, 4, 2, 4, 2, 1, 3, 2, 3, 2, 1, 8, 1, 8, 6, 7, 8, 6, 5, 7, 6, 8, 1, 3, 4, 3, 2, 3, 2, 3, 1, 8, 1, 2, 3, 2, 3, 4, 2, 1, 3, 1, 2, 3, 4, 5, 7, 6, 7, 6, 7, 5, 6, 5, 4, 3, 4, 2, 3, 1, 8, 7, 5, 6, 5, 7, 6, 8, 1, 2, 4, 2, 3, 4, 5, 6, 7, 8, 1, 2, 4, 2, 3, 1, 3, 2, 4, 5, 7, 8, 6, 8, 7, 6, 7, 6, 8, 7, 6, 5, 6, 5, 7, 6, 7, 5, 6, 8, 1, 8, 1, 8, 1, 2, 1, 8, 6, 8, 6, 7, 5, 4, 5, 4, 2, 3, 2, 3, 1, 3, 2, 3, 4, 3, 1, 8, 6, 7, 6, 5, 7, 5, 4, 3, 4, 3, 1, 3, 1, 2, 4, 2, 4, 3, 4, 2, 3, 4, 3, 2, 4, 3, 1, 2, 1, 3, 1, 2, 4, 2, 1, 2, 4, 2, 3, 4, 5, 6, 5, 7, 8, 7, 6, 8, 6, 7, 6, 7, 6, 7, 6, 8, 7, 8, 6, 8, 1, 3, 2, 4, 3, 4, 5, 6, 7, 6, 5, 4, 5, 7, 5, 6, 8, 1, 3, 1, 3, 1, 3, 1, 2, 3, 4, 2, 3, 1, 3, 1, 2, 1, 8, 7, 5, 6, 5, 4, 2, 3, 1, 8, 1, 3, 1, 2, 3, 1, 8, 1, 8, 6, 5, 4, 5, 6, 7, 6, 7, 8, 6, 8, 1, 8, 6, 8, 7, 5, 4, 3, 4, 5, 7, 6, 5, 6, 8, 1, 3, 4, 5, 4, 3, 1, 2, 3, 4, 2, 1, 2, 1, 3, 4, 5, 6, 7, 8, 1, 3, 2, 3, 1, 3, 2, 4, 5, 6, 5, 6, 5, 4, 2, 4, 5, 7, 6, 5, 7, 5, 4, 5, 4, 3, 1, 2, 3, 4, 3, 1, 2, 4, 2, 4, 5, 6, 7, 6, 7, 5, 6, 7, 5, 7, 6, 7, 6, 8, 6, 8, 7, 8, 7, 6, 8, 1, 2, 3, 2, 1, 2, 1, 2, 4, 2, 1, 2, 3, 1, 3, 2, 1, 8, 1, 3, 4, 3, 2, 4, 2, 1, 3, 1, 3, 4, 5, 7, 6, 7, 8, 7, 8, 6, 5, 7, 5, 4, 3, 4, 5, 4, 2, 3, 1, 8, 6, 5, 6, 7, 6, 5, 7, 6, 8, 6, 7, 6, 7, 5, 4, 3, 1, 3, 4, 5, 4, 3, 4, 5, 7, 6, 7, 8, 7, 8, 1, 8, 1, 2, 4, 3, 1, 2, 4, 3, 2, 1, 3, 1, 2, 4, 2, 4, 2, 3, 2, 4, 5, 7, 8, 1, 3, 4, 2, 1, 3, 2, 3, 4, 2, 4, 5, 7, 8, 1, 3, 1, 8, 1, 3, 4, 2, 1, 2, 1, 2, 4, 2, 1, 3, 4, 2, 1, 3, 2, 3, 2, 4, 5, 6, 8, 7, 6, 7, 6, 8, 6, 8, 6, 5, 4, 3, 1, 3, 4, 3, 2, 1, 2, 1, 3, 1, 8, 6, 8, 7, 6, 5, 4, 3, 4, 3, 4, 2, 3, 2, 3, 2, 3, 4, 3, 2, 3, 4, 5, 7, 5, 6, 8, 1, 3, 4, 2, 4, 5, 6, 5, 6, 8, 6, 7, 5, 6, 5, 7, 6, 7, 8, 1, 8, 6, 7, 5, 4, 5, 6, 7, 8, 1, 2, 3, 1, 8, 7, 6, 8, 6, 5, 6, 5, 7, 6, 5, 4, 3, 1, 2, 3, 4, 5, 7, 5, 7, 6, 7, 5, 6, 5, 4, 3, 2, 4, 3, 4, 5, 4, 5, 6, 5, 7, 5, 4, 3, 4, 3, 1, 3, 1, 3, 4, 5, 6, 7, 8, 1, 8, 7, 6, 5, 6, 8, 6, 8, 1, 8, 6, 7, 5, 7, 5, 7, 8, 1, 8, 1, 8, 7, 8, 1, 2, 4, 3, 1, 2, 3, 2, 1, 3, 4, 3, 2, 3, 2, 3, 2, 1, 8, 6, 5, 4, 3, 2, 4, 5, 7, 5, 4, 2, 3, 2, 4, 5, 7, 8, 6, 8, 7, 5, 6, 7, 6, 8, 6, 8, 7, 8, 1, 3, 2, 3, 4, 2, 3, 2, 3, 1, 2, 1, 8, 7, 6, 8, 1, 8, 6, 7, 8, 1, 8, 6, 8, 7, 8, 6, 8, 7, 6, 8, 6, 5, 4, 2, 3, 2, 1, 8, 7, 5, 4, 3, 4, 2, 4, 3, 1, 2, 1, 8, 7, 5, 6, 7, 8, 7, 5, 7, 6, 7, 8, 1, 3, 2, 4, 2, 4, 3, 4, 5, 4, 3, 2, 4, 5, 4, 3, 1, 3, 2, 1, 2, 3, 4, 5, 4, 3, 4, 2, 4, 3, 2, 1, 3, 4, 5, 7, 5, 4, 2, 3, 1, 8, 6, 7, 8, 6, 8, 6, 8, 7, 5, 6, 5, 7, 5, 4, 2, 4, 3, 2, 3, 1, 2, 1, 2, 3, 4, 5, 6, 7, 6, 8, 7, 8, 7, 6, 7, 6, 7, 5, 4, 5, 7, 5, 7, 5}
			} else if ss.runnum == 24 {
				seqid = [900]int{6, 7, 5, 4, 2, 3, 4, 2, 1, 3, 4, 2, 3, 1, 8, 6, 7, 5, 7, 6, 7, 8, 1, 3, 1, 8, 7, 6, 8, 1, 2, 3, 1, 2, 1, 8, 6, 5, 6, 8, 1, 2, 4, 3, 4, 2, 3, 1, 8, 1, 3, 1, 8, 7, 6, 5, 4, 2, 3, 1, 2, 1, 8, 6, 8, 7, 5, 6, 5, 6, 7, 5, 4, 3, 4, 3, 1, 3, 1, 3, 2, 4, 2, 3, 4, 3, 2, 4, 2, 1, 2, 1, 3, 4, 2, 1, 3, 1, 2, 4, 3, 1, 8, 7, 8, 1, 8, 6, 8, 7, 8, 6, 8, 6, 5, 7, 8, 6, 5, 6, 7, 6, 5, 7, 6, 8, 1, 3, 2, 4, 5, 7, 6, 7, 6, 5, 4, 2, 3, 1, 3, 4, 3, 2, 4, 5, 6, 5, 6, 8, 6, 5, 6, 8, 7, 8, 1, 3, 2, 3, 4, 5, 4, 3, 1, 8, 1, 3, 2, 4, 5, 4, 2, 1, 8, 6, 8, 6, 5, 7, 5, 4, 3, 2, 4, 5, 7, 5, 7, 8, 7, 6, 8, 1, 2, 3, 2, 1, 2, 4, 3, 1, 8, 6, 8, 6, 7, 8, 6, 8, 1, 2, 4, 5, 6, 8, 1, 3, 2, 4, 3, 2, 1, 3, 1, 3, 1, 2, 3, 4, 5, 6, 5, 4, 5, 7, 8, 1, 2, 4, 3, 1, 8, 6, 5, 4, 2, 1, 2, 3, 4, 2, 4, 2, 3, 2, 1, 3, 1, 2, 3, 2, 4, 3, 1, 8, 6, 7, 8, 1, 8, 6, 8, 7, 5, 7, 8, 6, 7, 5, 4, 3, 1, 8, 6, 5, 4, 3, 2, 1, 2, 1, 8, 6, 8, 6, 5, 7, 8, 7, 5, 6, 5, 6, 8, 7, 8, 1, 3, 4, 3, 1, 8, 1, 2, 4, 2, 1, 8, 1, 2, 4, 3, 4, 3, 4, 2, 3, 1, 8, 1, 8, 1, 8, 6, 5, 6, 7, 6, 7, 5, 4, 5, 4, 2, 1, 3, 1, 8, 1, 8, 6, 7, 6, 5, 4, 5, 6, 8, 1, 3, 2, 4, 5, 7, 5, 4, 3, 4, 3, 2, 4, 2, 4, 2, 3, 4, 3, 1, 2, 4, 3, 4, 5, 6, 7, 8, 1, 3, 4, 2, 1, 3, 2, 3, 2, 1, 3, 4, 2, 3, 2, 4, 3, 4, 5, 6, 5, 4, 5, 7, 6, 5, 4, 2, 3, 2, 1, 2, 3, 1, 3, 2, 3, 4, 2, 1, 3, 2, 3, 2, 4, 5, 4, 2, 1, 2, 4, 2, 1, 3, 4, 2, 3, 2, 4, 3, 1, 2, 3, 1, 2, 4, 5, 7, 5, 6, 8, 7, 6, 7, 6, 5, 7, 8, 6, 5, 6, 7, 8, 1, 3, 2, 4, 2, 1, 2, 4, 5, 7, 6, 7, 5, 6, 5, 6, 8, 7, 8, 6, 8, 6, 7, 8, 6, 8, 7, 5, 7, 8, 1, 3, 1, 8, 1, 2, 1, 3, 2, 3, 2, 4, 3, 1, 8, 7, 8, 1, 2, 3, 1, 3, 4, 3, 1, 8, 6, 8, 7, 5, 7, 8, 6, 8, 6, 7, 8, 7, 5, 4, 5, 4, 2, 1, 8, 6, 8, 7, 8, 7, 8, 1, 8, 7, 6, 8, 6, 5, 4, 2, 3, 2, 3, 2, 1, 8, 7, 8, 6, 5, 6, 5, 6, 8, 6, 5, 6, 7, 8, 6, 5, 6, 5, 4, 5, 6, 5, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 4, 2, 1, 2, 4, 3, 4, 3, 1, 8, 1, 3, 4, 5, 6, 5, 4, 2, 3, 1, 2, 4, 5, 6, 5, 4, 5, 7, 5, 4, 5, 6, 7, 8, 7, 8, 1, 3, 2, 1, 2, 4, 3, 4, 5, 6, 8, 6, 8, 7, 8, 6, 5, 6, 5, 4, 3, 1, 3, 4, 5, 4, 5, 6, 7, 8, 7, 8, 7, 6, 5, 6, 7, 8, 1, 2, 1, 2, 1, 2, 4, 2, 4, 2, 4, 2, 1, 8, 7, 8, 6, 8, 1, 2, 1, 3, 4, 2, 1, 8, 6, 5, 4, 3, 4, 5, 6, 7, 5, 7, 5, 6, 5, 6, 5, 4, 5, 6, 8, 7, 8, 6, 8, 7, 5, 4, 3, 1, 8, 6, 5, 4, 2, 1, 8, 6, 7, 8, 1, 8, 1, 3, 2, 1, 2, 4, 3, 4, 3, 2, 4, 2, 1, 3, 4, 3, 1, 3, 4, 2, 4, 2, 1, 3, 4, 2, 4, 3, 1, 8, 6, 7, 8, 7, 8, 6, 8, 1, 2, 1, 3, 4, 3, 1, 3, 4, 3, 4, 2, 4, 5, 6, 8, 7, 8, 6, 8, 1, 3, 2, 1, 3, 4, 5, 7, 6, 8, 6, 8, 7, 8, 7, 6, 5, 6, 8, 7, 8, 1, 2, 1, 8, 1, 2, 3, 4, 5, 4, 2, 3, 4, 3, 2, 4, 3, 1, 8, 1, 3, 1, 2, 3, 4, 2, 1, 3, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 7, 5, 7, 8, 6, 8, 1, 3, 4, 3, 1, 3, 1, 8, 6, 5, 4, 5, 4, 5, 6, 7, 6, 8, 7, 5, 6, 8, 7, 5, 6, 8, 1, 2, 1, 8, 6, 8, 7, 5, 4, 2, 4, 2, 3, 4, 3}
			} else if ss.runnum == 25 {
				seqid = [900]int{4, 2, 1, 3, 1, 8, 7, 5, 6, 7, 8, 1, 3, 2, 3, 1, 8, 6, 7, 8, 6, 5, 6, 5, 6, 7, 8, 1, 2, 4, 3, 1, 3, 4, 2, 3, 1, 8, 1, 2, 3, 2, 3, 4, 5, 4, 3, 1, 2, 4, 3, 1, 3, 2, 1, 3, 2, 4, 2, 4, 5, 4, 5, 6, 5, 6, 7, 6, 8, 6, 8, 6, 5, 4, 5, 6, 7, 6, 7, 8, 7, 8, 7, 6, 7, 5, 7, 8, 6, 8, 6, 7, 6, 8, 7, 8, 1, 2, 1, 8, 6, 7, 6, 7, 5, 6, 8, 1, 3, 1, 2, 1, 8, 6, 5, 7, 8, 7, 6, 8, 6, 5, 7, 6, 8, 6, 5, 4, 2, 4, 2, 3, 1, 3, 2, 1, 3, 1, 3, 1, 8, 1, 8, 1, 2, 1, 3, 4, 2, 4, 3, 4, 2, 1, 8, 6, 8, 7, 8, 6, 5, 6, 5, 4, 3, 2, 4, 3, 2, 1, 8, 6, 7, 8, 1, 3, 1, 8, 7, 6, 7, 5, 4, 5, 4, 2, 3, 2, 1, 8, 6, 5, 7, 5, 6, 5, 7, 5, 6, 7, 8, 1, 8, 1, 2, 3, 4, 2, 1, 3, 1, 2, 4, 3, 1, 3, 1, 8, 1, 3, 1, 3, 2, 3, 1, 3, 1, 2, 4, 5, 7, 5, 4, 5, 7, 5, 4, 2, 1, 2, 3, 2, 3, 1, 8, 1, 2, 1, 3, 4, 5, 7, 8, 6, 8, 7, 8, 7, 6, 5, 4, 3, 2, 4, 2, 4, 5, 7, 6, 8, 1, 2, 1, 3, 2, 1, 8, 6, 5, 4, 3, 1, 2, 4, 2, 4, 5, 4, 2, 4, 5, 6, 5, 4, 2, 4, 3, 4, 2, 1, 8, 1, 2, 1, 2, 1, 3, 1, 8, 1, 3, 1, 8, 6, 8, 1, 3, 4, 3, 1, 2, 1, 8, 7, 5, 6, 8, 1, 8, 1, 8, 1, 8, 6, 5, 7, 6, 5, 4, 3, 4, 5, 7, 6, 7, 6, 5, 6, 5, 7, 5, 4, 5, 6, 8, 1, 8, 1, 3, 2, 4, 5, 6, 5, 6, 7, 6, 7, 5, 6, 7, 6, 5, 7, 6, 8, 1, 2, 4, 3, 1, 8, 6, 8, 6, 5, 7, 6, 5, 6, 5, 7, 6, 5, 6, 7, 6, 7, 5, 4, 5, 4, 3, 1, 3, 4, 2, 3, 1, 8, 6, 7, 6, 7, 8, 7, 6, 8, 7, 6, 7, 6, 8, 7, 6, 8, 6, 8, 7, 8, 7, 8, 7, 5, 6, 8, 1, 8, 6, 7, 5, 7, 5, 6, 8, 6, 8, 6, 8, 6, 5, 6, 8, 1, 8, 6, 7, 5, 4, 5, 4, 2, 3, 2, 3, 2, 3, 4, 5, 4, 3, 4, 3, 1, 8, 6, 8, 7, 6, 7, 6, 5, 7, 5, 7, 8, 7, 5, 4, 3, 4, 5, 6, 7, 6, 8, 6, 8, 1, 3, 1, 3, 1, 2, 4, 2, 3, 2, 1, 8, 1, 8, 6, 8, 7, 8, 1, 2, 4, 3, 1, 3, 4, 2, 4, 5, 6, 7, 6, 7, 5, 6, 8, 6, 7, 5, 4, 3, 2, 4, 5, 4, 5, 4, 2, 3, 1, 8, 7, 5, 4, 3, 2, 1, 8, 7, 8, 1, 3, 4, 5, 6, 7, 6, 7, 5, 4, 5, 6, 5, 6, 5, 4, 3, 2, 1, 2, 4, 3, 2, 1, 3, 2, 1, 8, 1, 8, 7, 5, 4, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 8, 1, 3, 1, 8, 6, 8, 6, 8, 7, 8, 6, 8, 1, 2, 1, 2, 1, 2, 4, 2, 3, 1, 8, 7, 8, 6, 7, 8, 6, 8, 7, 6, 7, 5, 4, 5, 4, 2, 3, 1, 8, 7, 6, 5, 6, 7, 5, 4, 2, 3, 4, 2, 4, 5, 6, 5, 4, 2, 1, 2, 3, 2, 1, 2, 1, 3, 1, 2, 4, 3, 1, 8, 6, 7, 5, 6, 5, 7, 6, 5, 4, 5, 7, 6, 5, 4, 2, 1, 8, 7, 8, 1, 2, 4, 2, 3, 2, 4, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 4, 3, 1, 2, 4, 3, 2, 1, 8, 1, 2, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 3, 4, 5, 4, 2, 1, 3, 1, 3, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 2, 3, 2, 4, 5, 7, 8, 6, 5, 7, 6, 5, 6, 5, 7, 8, 1, 8, 1, 2, 4, 5, 4, 3, 4, 3, 1, 8, 7, 5, 7, 8, 6, 7, 8, 7, 5, 4, 5, 4, 5, 7, 5, 4, 3, 2, 4, 3, 4, 3, 1, 8, 6, 7, 5, 4, 5, 4, 2, 1, 8, 6, 7, 5, 4, 2, 1, 2, 1, 3, 2, 3, 1, 3, 2, 3, 4, 5, 6, 7, 8, 6, 8, 7, 6, 5, 7, 6, 8, 1, 3, 4, 2, 1, 2, 3, 2, 1, 8, 7, 8, 7, 5, 4, 5, 4, 2, 4, 5, 4, 5, 6, 7, 5, 4, 3, 4, 3, 1, 3, 1, 8, 1, 8, 1, 8, 7, 8, 1, 3, 1, 2, 1, 8, 1, 8, 6, 7, 8, 1, 8, 6, 7, 5, 6, 5, 4, 3, 1, 3}
			} else if ss.runnum == 26 {
				seqid = [900]int{1, 8, 7, 8, 7, 5, 6, 8, 1, 8, 1, 8, 1, 8, 1, 2, 1, 2, 3, 1, 2, 1, 2, 4, 5, 7, 8, 1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 1, 3, 2, 3, 4, 5, 4, 5, 4, 5, 6, 8, 7, 8, 7, 5, 7, 5, 6, 7, 6, 5, 4, 5, 4, 5, 6, 8, 6, 7, 8, 6, 7, 6, 5, 6, 8, 7, 5, 7, 6, 7, 8, 6, 5, 4, 2, 3, 1, 3, 4, 3, 4, 3, 4, 3, 2, 4, 2, 4, 5, 7, 5, 6, 7, 8, 6, 7, 6, 7, 8, 6, 8, 7, 8, 6, 5, 6, 7, 6, 5, 4, 5, 6, 8, 7, 6, 5, 7, 5, 7, 8, 6, 8, 1, 3, 4, 5, 4, 2, 3, 2, 1, 2, 1, 2, 3, 4, 3, 2, 3, 1, 3, 1, 8, 7, 5, 4, 5, 7, 6, 7, 6, 5, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3, 4, 2, 1, 2, 4, 3, 4, 2, 1, 8, 1, 8, 6, 7, 5, 4, 5, 6, 7, 8, 1, 3, 2, 4, 2, 3, 1, 8, 1, 3, 2, 3, 4, 2, 1, 8, 1, 8, 7, 5, 6, 5, 4, 3, 2, 4, 3, 2, 3, 1, 8, 6, 8, 1, 2, 1, 8, 7, 8, 7, 8, 1, 2, 4, 5, 7, 8, 7, 6, 5, 7, 8, 1, 2, 1, 8, 6, 5, 6, 8, 7, 5, 7, 6, 7, 5, 7, 6, 5, 6, 5, 4, 5, 7, 5, 4, 3, 2, 4, 5, 6, 8, 7, 5, 6, 8, 1, 2, 1, 3, 4, 5, 6, 5, 6, 7, 8, 1, 8, 6, 5, 4, 5, 7, 8, 7, 8, 7, 6, 8, 7, 6, 5, 6, 8, 1, 3, 4, 5, 4, 5, 6, 8, 6, 7, 5, 7, 8, 7, 8, 1, 2, 1, 8, 7, 5, 7, 6, 8, 1, 8, 6, 5, 6, 7, 6, 8, 7, 8, 6, 8, 1, 8, 6, 5, 4, 3, 1, 2, 3, 1, 2, 4, 3, 4, 3, 1, 8, 1, 8, 7, 8, 7, 8, 6, 5, 6, 7, 6, 7, 8, 6, 5, 4, 3, 4, 3, 1, 8, 7, 8, 1, 2, 4, 5, 4, 5, 4, 3, 1, 3, 1, 2, 3, 2, 4, 2, 3, 2, 1, 2, 4, 2, 3, 2, 1, 2, 4, 2, 1, 2, 1, 8, 7, 6, 7, 5, 6, 7, 5, 4, 3, 1, 2, 4, 2, 1, 2, 1, 3, 4, 2, 4, 5, 6, 5, 7, 5, 6, 8, 7, 8, 6, 8, 6, 7, 8, 1, 8, 6, 8, 6, 8, 1, 2, 4, 3, 1, 2, 4, 5, 6, 8, 6, 5, 4, 3, 1, 8, 6, 8, 6, 8, 6, 8, 1, 3, 1, 2, 1, 3, 4, 2, 3, 1, 8, 7, 5, 4, 5, 7, 5, 4, 3, 4, 2, 4, 5, 7, 5, 7, 5, 7, 6, 8, 6, 8, 7, 5, 4, 3, 4, 5, 7, 5, 6, 8, 6, 7, 5, 4, 3, 4, 3, 4, 5, 7, 5, 7, 6, 7, 8, 6, 7, 5, 6, 8, 1, 2, 1, 8, 1, 2, 1, 2, 3, 4, 3, 1, 8, 6, 5, 7, 6, 5, 4, 3, 1, 3, 2, 4, 5, 6, 5, 4, 5, 6, 7, 8, 1, 2, 3, 1, 3, 4, 2, 1, 2, 3, 2, 1, 8, 7, 8, 1, 2, 4, 5, 4, 3, 1, 8, 6, 8, 7, 5, 4, 5, 7, 6, 7, 6, 8, 1, 8, 6, 5, 6, 8, 1, 2, 1, 8, 6, 7, 5, 6, 5, 6, 7, 8, 6, 7, 6, 7, 5, 6, 7, 6, 5, 7, 5, 7, 8, 7, 8, 6, 5, 6, 7, 6, 8, 1, 8, 7, 8, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 4, 5, 6, 8, 7, 8, 7, 6, 5, 6, 5, 6, 5, 6, 7, 6, 7, 6, 5, 7, 8, 7, 5, 6, 7, 6, 7, 6, 8, 7, 8, 1, 2, 4, 3, 1, 8, 7, 5, 6, 5, 6, 8, 7, 8, 6, 7, 5, 7, 8, 7, 8, 1, 8, 7, 6, 7, 6, 7, 5, 4, 3, 1, 3, 4, 5, 7, 8, 7, 6, 5, 7, 8, 6, 5, 4, 2, 3, 1, 2, 3, 4, 5, 6, 5, 6, 7, 5, 4, 2, 4, 5, 6, 7, 8, 6, 7, 5, 6, 8, 1, 2, 3, 2, 1, 8, 6, 8, 1, 8, 1, 8, 7, 6, 5, 4, 5, 4, 2, 4, 2, 1, 2, 4, 5, 7, 5, 7, 8, 1, 2, 1, 3, 2, 3, 1, 8, 6, 8, 6, 5, 6, 7, 8, 7, 8, 7, 8, 6, 5, 6, 8, 6, 8, 1, 2, 4, 5, 6, 7, 5, 6, 5, 7, 5, 6, 8, 7, 5, 6, 8, 6, 8, 6, 7, 8, 6, 8, 1, 3, 4, 2, 1, 8, 7, 5, 6, 7, 5, 4, 3, 1, 2, 4, 2, 3, 4, 5, 6, 7, 5, 4, 5, 4, 5, 4, 3, 4, 3, 4, 3, 2, 4, 2, 3, 1, 3, 1, 8, 7, 5, 6, 8, 7, 6, 5, 7, 8, 1, 8, 7, 5, 6, 5, 7, 8, 6, 8, 1, 3, 1, 2, 4}
			} else if ss.runnum == 27 {
				seqid = [900]int{1, 8, 1, 3, 1, 3, 2, 3, 4, 2, 1, 8, 6, 8, 1, 8, 1, 3, 2, 3, 4, 5, 7, 5, 4, 2, 3, 1, 2, 1, 2, 3, 2, 4, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 1, 8, 6, 8, 6, 8, 6, 5, 4, 2, 4, 3, 4, 3, 4, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 8, 7, 6, 8, 7, 6, 7, 8, 6, 5, 4, 3, 1, 2, 1, 3, 4, 3, 2, 4, 2, 4, 2, 1, 2, 4, 3, 2, 4, 2, 4, 3, 4, 2, 3, 1, 2, 4, 3, 1, 8, 1, 8, 1, 3, 4, 2, 1, 2, 4, 2, 4, 2, 3, 4, 3, 1, 8, 1, 2, 4, 3, 2, 1, 2, 3, 2, 3, 2, 1, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 5, 7, 8, 1, 2, 4, 5, 4, 3, 1, 2, 1, 2, 4, 2, 4, 3, 4, 3, 1, 2, 3, 1, 8, 7, 8, 7, 6, 5, 7, 8, 7, 8, 7, 8, 1, 8, 1, 2, 4, 3, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 5, 4, 3, 4, 2, 3, 1, 3, 2, 1, 2, 3, 1, 2, 3, 2, 3, 4, 5, 6, 8, 7, 6, 8, 1, 8, 1, 2, 3, 4, 3, 1, 3, 4, 5, 4, 2, 4, 5, 7, 5, 7, 5, 4, 5, 4, 3, 4, 3, 2, 1, 8, 1, 8, 7, 8, 7, 8, 7, 6, 5, 4, 3, 4, 3, 2, 1, 8, 1, 3, 4, 2, 1, 2, 4, 2, 1, 2, 1, 8, 1, 2, 1, 3, 2, 3, 2, 1, 3, 4, 2, 1, 2, 1, 3, 4, 3, 2, 1, 3, 1, 8, 1, 3, 4, 2, 1, 2, 3, 2, 3, 1, 2, 4, 2, 4, 5, 6, 8, 1, 8, 1, 3, 2, 1, 3, 4, 2, 3, 2, 3, 2, 1, 8, 7, 6, 8, 6, 8, 6, 7, 6, 7, 5, 4, 3, 2, 4, 3, 4, 3, 1, 8, 6, 8, 6, 8, 6, 8, 1, 3, 1, 8, 6, 8, 6, 7, 8, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 2, 4, 2, 4, 2, 3, 1, 2, 3, 4, 3, 2, 1, 2, 4, 5, 4, 3, 4, 3, 2, 4, 3, 2, 3, 2, 4, 2, 3, 2, 3, 1, 2, 1, 3, 2, 1, 8, 7, 5, 4, 2, 3, 4, 5, 7, 6, 5, 6, 5, 7, 8, 7, 8, 1, 8, 6, 7, 5, 4, 3, 2, 3, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 3, 2, 4, 5, 7, 6, 8, 6, 8, 7, 6, 8, 7, 6, 8, 1, 8, 1, 3, 4, 2, 3, 4, 2, 4, 2, 3, 1, 8, 7, 6, 7, 5, 7, 6, 8, 7, 6, 7, 6, 5, 6, 5, 6, 5, 4, 3, 1, 3, 2, 1, 3, 1, 3, 1, 3, 2, 4, 2, 1, 8, 1, 2, 3, 2, 3, 2, 3, 1, 8, 6, 7, 5, 7, 5, 7, 6, 5, 6, 8, 7, 6, 5, 4, 5, 7, 8, 1, 8, 6, 8, 6, 5, 4, 3, 4, 2, 3, 1, 2, 3, 4, 2, 3, 1, 8, 1, 8, 6, 8, 7, 6, 7, 8, 7, 6, 7, 6, 8, 7, 5, 4, 3, 4, 2, 3, 2, 3, 1, 2, 3, 4, 3, 4, 2, 3, 4, 3, 4, 2, 1, 2, 1, 2, 4, 2, 4, 3, 2, 3, 2, 4, 2, 3, 2, 3, 4, 2, 3, 4, 2, 3, 1, 3, 4, 3, 2, 3, 2, 3, 4, 5, 6, 5, 7, 5, 7, 8, 1, 8, 6, 8, 1, 3, 2, 4, 3, 4, 2, 3, 1, 2, 4, 2, 3, 4, 2, 3, 1, 8, 1, 8, 1, 2, 1, 3, 1, 2, 4, 5, 6, 7, 6, 8, 7, 5, 4, 5, 7, 8, 7, 5, 4, 2, 3, 1, 3, 4, 3, 4, 2, 1, 2, 1, 8, 6, 7, 8, 1, 3, 1, 8, 7, 5, 6, 5, 7, 6, 5, 7, 8, 6, 5, 6, 5, 4, 3, 2, 3, 1, 8, 7, 8, 7, 6, 8, 1, 2, 4, 5, 4, 3, 1, 2, 3, 1, 3, 2, 3, 1, 3, 4, 3, 2, 3, 4, 3, 4, 2, 1, 8, 1, 3, 1, 3, 2, 1, 2, 4, 5, 6, 5, 6, 8, 1, 2, 4, 2, 4, 3, 2, 4, 5, 4, 2, 4, 3, 1, 3, 4, 5, 7, 8, 6, 7, 6, 7, 6, 7, 5, 4, 3, 1, 2, 4, 3, 1, 8, 1, 8, 7, 6, 8, 1, 3, 4, 3, 4, 3, 1, 8, 1, 3, 4, 5, 6, 8, 1, 3, 4, 2, 3, 1, 3, 1, 8, 6, 7, 8, 7, 5, 7, 6, 7, 8, 7, 5, 4, 5, 4, 5, 7, 8, 1, 2, 1, 8, 7, 6, 8, 1, 2, 3, 2, 3, 4, 3, 2, 1, 3, 4, 3, 2, 1, 2, 1, 2, 4, 5, 4, 2, 1, 2, 3, 2, 3, 1, 3, 2, 1, 3, 2, 1, 3, 2, 4, 2, 4, 5, 7, 5, 6, 8, 1, 3, 4, 2, 1, 3, 2, 1, 3, 1, 3, 4, 2}
			} else if ss.runnum == 28 {
				seqid = [900]int{5, 7, 5, 7, 5, 7, 5, 6, 8, 6, 7, 8, 7, 8, 6, 8, 7, 6, 7, 8, 7, 6, 8, 6, 8, 7, 6, 7, 6, 7, 6, 7, 6, 8, 7, 8, 7, 6, 7, 6, 5, 4, 2, 1, 2, 3, 4, 2, 4, 2, 3, 2, 4, 5, 6, 5, 7, 8, 7, 5, 4, 2, 4, 5, 4, 3, 2, 4, 5, 4, 3, 4, 3, 4, 5, 4, 3, 2, 3, 2, 4, 5, 6, 5, 7, 8, 1, 2, 4, 2, 3, 2, 3, 4, 3, 2, 3, 1, 3, 1, 8, 6, 7, 5, 4, 3, 4, 3, 4, 2, 4, 2, 3, 2, 1, 8, 1, 2, 4, 3, 1, 2, 1, 8, 1, 3, 2, 3, 1, 3, 1, 2, 1, 2, 4, 2, 4, 3, 4, 5, 6, 8, 7, 8, 1, 8, 1, 2, 1, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 8, 6, 8, 1, 2, 3, 2, 4, 3, 1, 2, 4, 2, 4, 3, 4, 5, 6, 8, 6, 7, 8, 7, 8, 7, 5, 7, 8, 1, 3, 1, 2, 4, 5, 6, 5, 6, 5, 4, 5, 7, 8, 1, 2, 3, 4, 2, 4, 5, 7, 5, 7, 5, 6, 5, 6, 8, 6, 7, 8, 1, 8, 6, 8, 7, 6, 8, 1, 8, 7, 8, 6, 5, 7, 5, 7, 6, 8, 7, 5, 6, 8, 7, 5, 6, 8, 6, 8, 6, 8, 7, 5, 4, 3, 1, 2, 1, 3, 1, 3, 2, 3, 2, 1, 3, 1, 2, 1, 2, 3, 4, 3, 4, 5, 7, 5, 4, 5, 6, 7, 5, 6, 7, 6, 8, 7, 6, 5, 7, 6, 7, 5, 7, 6, 7, 5, 7, 5, 6, 8, 1, 2, 1, 8, 1, 8, 6, 5, 4, 5, 6, 5, 7, 5, 4, 2, 1, 2, 4, 2, 1, 3, 2, 1, 3, 4, 3, 4, 2, 4, 5, 7, 8, 6, 8, 7, 5, 6, 7, 5, 6, 7, 5, 4, 3, 4, 2, 4, 3, 1, 3, 1, 2, 1, 3, 2, 4, 2, 1, 8, 1, 2, 3, 1, 2, 4, 5, 4, 5, 7, 5, 7, 6, 8, 6, 7, 5, 7, 5, 7, 5, 4, 2, 3, 4, 3, 1, 3, 4, 2, 3, 2, 1, 3, 2, 4, 5, 6, 7, 5, 4, 3, 2, 4, 3, 4, 5, 4, 2, 4, 3, 4, 3, 2, 1, 8, 7, 6, 8, 6, 7, 6, 8, 7, 6, 8, 1, 2, 3, 1, 8, 7, 5, 4, 2, 4, 3, 1, 3, 1, 8, 6, 8, 7, 6, 7, 6, 8, 1, 2, 3, 2, 3, 2, 3, 4, 3, 4, 5, 6, 7, 5, 7, 8, 1, 8, 6, 8, 6, 8, 7, 6, 8, 6, 8, 1, 3, 2, 1, 3, 2, 3, 2, 1, 3, 2, 1, 3, 2, 3, 1, 2, 1, 8, 6, 5, 6, 5, 7, 6, 5, 7, 8, 7, 5, 7, 5, 4, 3, 1, 8, 1, 2, 3, 1, 8, 1, 3, 4, 5, 4, 2, 1, 3, 4, 2, 4, 2, 3, 2, 1, 8, 6, 5, 7, 6, 7, 5, 4, 5, 7, 6, 7, 6, 5, 4, 2, 1, 2, 3, 2, 4, 3, 4, 3, 1, 3, 1, 3, 2, 3, 1, 8, 6, 5, 6, 5, 7, 5, 4, 5, 7, 6, 8, 6, 5, 6, 5, 4, 3, 1, 3, 4, 5, 4, 5, 7, 5, 7, 5, 6, 8, 6, 5, 4, 2, 3, 1, 2, 3, 2, 1, 2, 1, 8, 7, 6, 7, 8, 1, 8, 1, 2, 3, 4, 2, 3, 4, 2, 1, 3, 2, 1, 3, 1, 8, 7, 8, 6, 8, 6, 8, 7, 8, 6, 7, 5, 4, 3, 2, 3, 2, 4, 3, 2, 4, 3, 2, 1, 2, 1, 2, 4, 2, 4, 2, 3, 4, 3, 1, 8, 7, 6, 5, 7, 6, 7, 8, 7, 5, 6, 7, 8, 6, 5, 6, 8, 7, 5, 7, 8, 7, 6, 7, 6, 7, 8, 7, 8, 1, 3, 4, 5, 6, 8, 7, 5, 7, 6, 7, 8, 1, 2, 4, 2, 1, 3, 4, 5, 4, 2, 4, 3, 4, 2, 3, 1, 3, 1, 2, 4, 2, 3, 1, 3, 4, 5, 4, 5, 6, 5, 7, 8, 7, 8, 1, 8, 6, 8, 1, 3, 2, 4, 5, 7, 6, 7, 8, 1, 3, 1, 3, 2, 1, 3, 4, 5, 4, 2, 4, 2, 1, 8, 6, 8, 1, 3, 2, 1, 3, 4, 3, 4, 2, 4, 2, 4, 3, 2, 4, 5, 6, 7, 6, 7, 5, 7, 5, 4, 2, 4, 5, 6, 8, 6, 8, 6, 8, 6, 8, 1, 8, 7, 8, 1, 2, 4, 2, 3, 1, 3, 2, 1, 3, 4, 5, 6, 8, 6, 8, 1, 8, 1, 3, 2, 4, 5, 4, 2, 4, 3, 1, 3, 1, 2, 1, 8, 1, 3, 4, 3, 4, 5, 6, 8, 7, 5, 4, 2, 4, 5, 7, 8, 7, 8, 6, 8, 6, 7, 8, 7, 6, 5, 6, 8, 7, 8, 7, 6, 5, 6, 8, 1, 8, 6, 8, 1, 3, 4, 2, 4, 3, 4, 2, 4, 5, 6, 8, 6, 8, 6, 5, 4, 2, 4, 2, 4, 3, 1, 2, 4}
			} else if ss.runnum == 29 {
				seqid = [900]int{7, 5, 6, 7, 8, 1, 8, 7, 5, 6, 5, 6, 5, 4, 5, 4, 5, 7, 5, 4, 5, 6, 5, 7, 6, 7, 6, 5, 6, 5, 4, 3, 4, 5, 6, 8, 6, 7, 8, 1, 8, 1, 3, 4, 5, 4, 5, 7, 8, 1, 2, 3, 2, 1, 8, 1, 2, 1, 8, 6, 7, 6, 7, 8, 1, 2, 1, 2, 3, 1, 3, 4, 3, 2, 4, 3, 2, 4, 5, 7, 5, 7, 5, 6, 7, 8, 6, 7, 6, 8, 7, 5, 7, 8, 1, 2, 4, 2, 3, 2, 4, 5, 4, 5, 6, 7, 8, 6, 7, 6, 7, 6, 5, 4, 3, 4, 2, 3, 2, 1, 3, 4, 5, 6, 7, 8, 6, 5, 4, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 8, 6, 8, 7, 6, 5, 6, 5, 4, 5, 6, 8, 7, 8, 1, 8, 7, 5, 7, 6, 7, 6, 5, 6, 5, 4, 2, 3, 4, 2, 4, 3, 4, 2, 4, 5, 7, 6, 8, 1, 3, 2, 4, 5, 6, 5, 7, 8, 1, 8, 1, 3, 1, 8, 7, 8, 1, 3, 4, 2, 1, 8, 1, 2, 4, 2, 1, 2, 4, 2, 4, 2, 3, 1, 8, 1, 8, 1, 8, 7, 5, 6, 8, 1, 3, 4, 5, 6, 8, 6, 7, 8, 6, 5, 6, 8, 1, 3, 1, 2, 1, 3, 4, 2, 4, 5, 7, 8, 7, 6, 7, 6, 8, 6, 5, 7, 5, 7, 6, 8, 6, 8, 7, 5, 4, 2, 1, 8, 1, 2, 1, 3, 4, 5, 7, 6, 7, 8, 1, 3, 4, 2, 3, 1, 3, 2, 4, 2, 3, 1, 2, 4, 2, 4, 3, 1, 2, 4, 3, 4, 3, 1, 3, 1, 3, 4, 5, 7, 6, 8, 1, 8, 1, 3, 2, 1, 2, 1, 2, 4, 3, 2, 4, 3, 1, 3, 2, 3, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 2, 4, 5, 4, 3, 4, 5, 7, 5, 7, 5, 4, 5, 6, 8, 1, 2, 3, 4, 3, 2, 4, 3, 1, 2, 4, 2, 4, 3, 1, 3, 2, 4, 2, 4, 5, 6, 5, 7, 6, 5, 6, 7, 8, 6, 7, 6, 5, 7, 6, 8, 7, 8, 6, 5, 6, 5, 4, 3, 2, 3, 4, 2, 4, 5, 6, 5, 6, 5, 4, 3, 4, 3, 4, 3, 1, 3, 4, 3, 1, 8, 6, 5, 6, 7, 6, 5, 7, 6, 5, 7, 6, 5, 6, 7, 5, 4, 3, 2, 1, 2, 3, 2, 4, 2, 4, 2, 1, 3, 1, 2, 4, 5, 7, 6, 8, 1, 8, 7, 6, 5, 6, 7, 8, 6, 7, 8, 1, 2, 3, 1, 2, 1, 8, 7, 6, 7, 8, 1, 8, 6, 8, 7, 5, 6, 7, 8, 1, 2, 4, 2, 1, 3, 2, 1, 2, 1, 2, 1, 3, 4, 2, 4, 5, 7, 5, 6, 5, 6, 7, 5, 6, 8, 1, 3, 4, 3, 2, 4, 2, 1, 3, 4, 5, 6, 7, 8, 7, 6, 7, 8, 1, 8, 6, 5, 7, 8, 1, 3, 4, 2, 1, 3, 1, 8, 1, 2, 3, 1, 2, 4, 5, 6, 7, 5, 7, 6, 5, 4, 2, 3, 1, 3, 1, 3, 1, 3, 2, 1, 3, 4, 5, 6, 8, 7, 6, 8, 1, 8, 6, 7, 6, 5, 4, 2, 4, 5, 7, 5, 4, 3, 2, 4, 2, 4, 3, 2, 4, 2, 3, 2, 3, 2, 1, 3, 4, 2, 3, 2, 4, 5, 4, 3, 1, 8, 7, 6, 8, 7, 6, 7, 6, 8, 1, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 3, 4, 3, 1, 2, 4, 5, 4, 2, 3, 2, 3, 2, 3, 4, 5, 7, 8, 1, 3, 2, 3, 1, 8, 7, 8, 7, 8, 7, 8, 6, 7, 6, 7, 8, 6, 5, 4, 5, 7, 8, 6, 7, 6, 7, 5, 4, 3, 1, 2, 4, 5, 4, 5, 6, 7, 5, 4, 2, 4, 5, 7, 5, 7, 8, 7, 5, 7, 5, 4, 3, 1, 8, 1, 2, 1, 8, 1, 2, 4, 2, 3, 2, 1, 3, 1, 3, 4, 2, 1, 3, 4, 2, 1, 8, 1, 3, 2, 1, 8, 7, 5, 4, 5, 4, 3, 4, 3, 2, 4, 5, 4, 3, 2, 3, 1, 8, 1, 3, 4, 5, 6, 8, 1, 8, 6, 7, 8, 6, 7, 6, 8, 1, 8, 7, 6, 5, 4, 5, 7, 8, 6, 5, 6, 5, 7, 6, 5, 6, 5, 6, 7, 8, 6, 8, 7, 5, 7, 6, 8, 7, 5, 6, 8, 7, 8, 6, 5, 7, 6, 7, 5, 6, 7, 8, 6, 7, 6, 8, 7, 5, 7, 5, 4, 2, 4, 5, 6, 5, 4, 2, 3, 2, 3, 1, 2, 1, 8, 7, 6, 7, 8, 7, 5, 7, 6, 5, 6, 7, 8, 7, 6, 7, 8, 6, 7, 8, 7, 5, 4, 5, 4, 2, 3, 4, 5, 7, 6, 8, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 5, 7, 8, 7, 8, 6, 8, 7, 6, 8, 7, 5, 6, 8, 6, 5, 7, 8, 7, 8, 6, 8, 7, 5, 4, 5, 4, 3, 2}
			} else if ss.runnum == 30 {
				seqid = [900]int{6, 5, 4, 5, 7, 8, 6, 5, 7, 8, 7, 5, 6, 5, 4, 5, 6, 5, 6, 8, 1, 2, 4, 2, 3, 1, 3, 1, 8, 7, 6, 7, 8, 6, 5, 7, 8, 7, 8, 1, 2, 4, 5, 6, 8, 7, 6, 5, 7, 5, 6, 7, 5, 7, 5, 6, 8, 7, 6, 5, 6, 8, 1, 3, 2, 1, 3, 4, 2, 4, 3, 4, 5, 7, 8, 7, 8, 7, 8, 1, 8, 1, 2, 4, 3, 2, 1, 8, 6, 7, 8, 1, 2, 3, 4, 5, 7, 8, 6, 5, 4, 5, 4, 2, 3, 4, 5, 7, 8, 7, 6, 8, 6, 5, 7, 6, 5, 6, 8, 7, 5, 4, 3, 2, 1, 3, 4, 5, 4, 5, 7, 8, 1, 2, 3, 2, 1, 3, 1, 8, 7, 5, 4, 2, 3, 4, 5, 7, 6, 5, 7, 5, 7, 8, 1, 2, 3, 2, 3, 1, 2, 3, 2, 3, 1, 8, 1, 8, 6, 5, 4, 2, 3, 2, 3, 4, 3, 4, 5, 7, 6, 5, 6, 5, 4, 2, 1, 3, 2, 1, 3, 1, 3, 4, 5, 6, 7, 5, 6, 7, 6, 8, 1, 2, 3, 2, 3, 4, 2, 1, 8, 7, 5, 4, 3, 1, 3, 2, 1, 3, 1, 8, 1, 2, 3, 4, 5, 7, 5, 7, 8, 1, 2, 1, 3, 4, 3, 1, 3, 2, 4, 3, 4, 5, 7, 8, 1, 2, 3, 2, 4, 3, 1, 2, 3, 2, 4, 5, 6, 7, 8, 6, 5, 4, 3, 2, 4, 5, 4, 2, 3, 2, 3, 2, 3, 1, 2, 3, 1, 2, 1, 8, 6, 8, 7, 6, 8, 6, 8, 7, 6, 7, 6, 7, 5, 7, 6, 7, 6, 5, 7, 8, 6, 7, 8, 7, 8, 6, 5, 7, 8, 7, 8, 6, 5, 6, 8, 7, 8, 6, 8, 6, 8, 6, 8, 7, 6, 5, 7, 8, 6, 8, 7, 8, 1, 8, 6, 8, 1, 8, 7, 5, 4, 2, 1, 2, 4, 3, 2, 3, 1, 8, 6, 7, 6, 5, 6, 5, 4, 3, 2, 3, 2, 4, 3, 4, 3, 1, 2, 1, 2, 1, 3, 4, 3, 2, 3, 4, 5, 4, 3, 1, 3, 2, 1, 2, 4, 5, 7, 5, 6, 8, 1, 2, 1, 8, 7, 6, 5, 7, 5, 6, 7, 5, 6, 5, 6, 5, 7, 8, 1, 3, 1, 3, 4, 5, 7, 5, 7, 8, 1, 2, 1, 3, 1, 3, 1, 3, 2, 4, 2, 3, 1, 2, 1, 3, 4, 2, 4, 3, 4, 3, 1, 8, 1, 3, 2, 4, 2, 3, 2, 1, 8, 7, 8, 6, 5, 7, 6, 5, 4, 5, 7, 8, 6, 7, 6, 5, 7, 6, 8, 7, 5, 4, 2, 1, 3, 1, 8, 7, 8, 1, 3, 4, 3, 2, 1, 2, 3, 1, 2, 1, 2, 1, 8, 6, 7, 5, 4, 5, 7, 8, 7, 5, 7, 5, 7, 6, 7, 5, 7, 5, 7, 5, 6, 5, 7, 8, 1, 8, 1, 8, 6, 8, 7, 6, 7, 6, 5, 6, 8, 7, 6, 7, 5, 6, 8, 6, 5, 4, 3, 4, 5, 4, 3, 4, 5, 6, 8, 6, 7, 8, 6, 5, 7, 6, 8, 6, 7, 8, 1, 8, 6, 8, 1, 3, 4, 5, 7, 5, 7, 5, 4, 5, 7, 8, 1, 3, 4, 5, 4, 3, 1, 3, 4, 2, 1, 8, 1, 8, 7, 5, 4, 3, 2, 3, 4, 2, 4, 5, 7, 8, 1, 8, 1, 8, 6, 5, 7, 5, 4, 2, 3, 4, 3, 1, 3, 1, 8, 1, 8, 6, 8, 7, 5, 6, 7, 8, 6, 8, 1, 2, 4, 3, 4, 3, 2, 1, 2, 1, 3, 4, 2, 3, 2, 3, 1, 8, 7, 6, 8, 7, 8, 1, 2, 3, 1, 2, 1, 2, 4, 2, 3, 2, 4, 5, 6, 7, 5, 7, 8, 6, 5, 4, 5, 6, 7, 8, 7, 8, 7, 6, 8, 6, 5, 6, 5, 6, 7, 6, 7, 5, 7, 6, 8, 6, 5, 7, 5, 7, 6, 7, 8, 7, 8, 1, 2, 1, 8, 1, 8, 1, 8, 7, 5, 7, 6, 8, 7, 6, 8, 1, 3, 2, 3, 2, 4, 5, 7, 5, 7, 8, 6, 5, 6, 5, 6, 8, 6, 7, 6, 7, 5, 7, 6, 8, 7, 5, 6, 8, 1, 8, 1, 8, 6, 7, 8, 7, 8, 7, 5, 7, 8, 7, 6, 5, 4, 2, 1, 3, 2, 3, 2, 4, 5, 4, 3, 2, 3, 1, 2, 3, 2, 1, 8, 6, 5, 4, 5, 7, 6, 8, 6, 8, 6, 7, 5, 4, 2, 4, 2, 3, 4, 3, 4, 2, 4, 2, 1, 3, 1, 2, 3, 2, 3, 4, 3, 1, 2, 1, 3, 4, 2, 1, 3, 1, 8, 6, 7, 6, 5, 6, 8, 1, 8, 7, 5, 7, 6, 7, 5, 6, 7, 8, 7, 8, 7, 5, 7, 8, 1, 8, 6, 5, 4, 3, 4, 3, 1, 3, 1, 2, 1, 8, 7, 8, 1, 8, 7, 6, 7, 5, 6, 8, 1, 3, 1, 8, 6, 7, 6, 5, 4, 3, 4, 2, 4, 2, 4, 2, 1, 8, 6, 8, 7, 5, 7, 8, 6, 7}
			} else if ss.runnum == 31 {
				seqid = [900]int{2, 1, 8, 1, 8, 7, 6, 5, 4, 5, 7, 8, 6, 8, 1, 8, 7, 8, 1, 3, 4, 2, 4, 2, 1, 2, 3, 2, 4, 3, 1, 8, 7, 5, 6, 8, 1, 3, 4, 5, 6, 8, 7, 5, 4, 5, 6, 7, 6, 8, 1, 3, 4, 5, 4, 3, 2, 3, 4, 2, 3, 4, 3, 4, 5, 4, 3, 2, 3, 2, 1, 3, 2, 1, 2, 3, 1, 8, 6, 5, 4, 3, 4, 2, 1, 8, 7, 8, 7, 8, 6, 8, 1, 2, 1, 8, 1, 8, 1, 2, 1, 8, 1, 2, 3, 4, 3, 1, 2, 3, 2, 1, 8, 6, 7, 8, 7, 6, 8, 1, 2, 3, 1, 3, 4, 2, 1, 3, 4, 2, 4, 3, 1, 8, 7, 8, 1, 2, 4, 3, 4, 3, 2, 3, 2, 1, 8, 1, 8, 7, 5, 7, 6, 5, 4, 2, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 4, 3, 4, 2, 1, 3, 1, 3, 1, 2, 4, 3, 4, 2, 3, 1, 2, 1, 8, 1, 8, 6, 5, 7, 8, 7, 5, 4, 5, 4, 3, 2, 1, 3, 4, 3, 1, 8, 6, 8, 6, 8, 1, 3, 4, 5, 7, 8, 1, 3, 1, 3, 4, 5, 7, 6, 5, 6, 8, 1, 8, 1, 8, 6, 5, 4, 3, 4, 5, 7, 5, 4, 2, 3, 2, 3, 1, 3, 2, 1, 8, 6, 7, 5, 7, 5, 7, 6, 8, 1, 2, 3, 4, 5, 6, 5, 7, 8, 7, 6, 5, 6, 5, 6, 5, 4, 3, 4, 3, 2, 3, 4, 3, 4, 5, 7, 5, 7, 6, 5, 7, 5, 6, 5, 4, 5, 4, 3, 1, 2, 4, 3, 1, 3, 2, 4, 5, 7, 5, 6, 5, 6, 7, 6, 8, 6, 7, 6, 5, 6, 5, 7, 8, 7, 6, 8, 1, 3, 2, 1, 8, 1, 2, 1, 3, 4, 5, 4, 3, 4, 5, 4, 2, 4, 5, 6, 5, 7, 5, 7, 5, 6, 5, 6, 7, 5, 7, 5, 4, 5, 4, 5, 7, 5, 7, 6, 8, 6, 5, 7, 6, 5, 6, 8, 7, 6, 8, 1, 8, 6, 5, 4, 3, 4, 5, 4, 3, 1, 8, 6, 7, 6, 8, 6, 8, 6, 7, 8, 1, 2, 3, 2, 1, 3, 4, 2, 4, 3, 2, 4, 2, 3, 1, 3, 4, 3, 1, 2, 4, 3, 1, 3, 1, 8, 7, 6, 7, 8, 6, 7, 6, 7, 6, 7, 5, 7, 6, 5, 7, 6, 7, 6, 8, 7, 5, 4, 2, 4, 5, 6, 8, 6, 7, 6, 8, 1, 2, 4, 5, 4, 3, 2, 3, 4, 2, 3, 4, 2, 4, 3, 4, 2, 3, 4, 3, 1, 3, 2, 3, 2, 3, 4, 2, 4, 5, 4, 2, 4, 2, 4, 5, 6, 5, 6, 8, 6, 5, 6, 7, 8, 6, 5, 4, 3, 2, 4, 2, 4, 5, 4, 2, 1, 3, 2, 1, 8, 1, 3, 1, 8, 7, 5, 6, 7, 5, 4, 5, 4, 2, 1, 8, 6, 7, 5, 7, 8, 7, 6, 8, 6, 8, 6, 5, 7, 6, 7, 8, 1, 8, 6, 7, 6, 7, 8, 7, 8, 1, 3, 2, 1, 2, 3, 1, 3, 1, 3, 4, 5, 4, 5, 7, 8, 7, 8, 1, 3, 4, 2, 3, 1, 8, 7, 8, 1, 2, 4, 2, 1, 3, 1, 8, 1, 8, 1, 2, 3, 2, 3, 1, 8, 1, 8, 6, 5, 6, 5, 4, 5, 6, 8, 6, 8, 6, 8, 1, 2, 4, 2, 4, 2, 4, 2, 4, 3, 4, 2, 1, 8, 1, 8, 7, 8, 7, 8, 6, 8, 7, 5, 6, 7, 6, 5, 4, 3, 1, 2, 1, 8, 1, 3, 2, 4, 5, 7, 6, 5, 7, 8, 6, 7, 5, 6, 7, 8, 6, 5, 7, 8, 7, 6, 8, 1, 8, 7, 5, 7, 6, 5, 4, 3, 1, 8, 7, 8, 7, 5, 4, 3, 1, 2, 1, 3, 2, 4, 3, 2, 4, 3, 4, 5, 4, 5, 4, 2, 4, 5, 4, 2, 4, 5, 7, 5, 7, 6, 7, 6, 5, 6, 5, 7, 5, 4, 2, 3, 2, 4, 2, 4, 2, 3, 2, 3, 1, 2, 4, 5, 7, 6, 8, 7, 6, 5, 7, 8, 1, 3, 4, 5, 7, 6, 5, 4, 5, 4, 5, 6, 5, 7, 8, 7, 5, 7, 8, 6, 5, 7, 6, 7, 6, 8, 6, 8, 6, 8, 1, 3, 4, 5, 6, 7, 6, 7, 5, 4, 3, 4, 5, 7, 6, 8, 6, 5, 4, 5, 4, 3, 2, 1, 2, 4, 2, 4, 2, 1, 3, 4, 5, 6, 8, 6, 8, 7, 8, 6, 5, 6, 8, 1, 2, 3, 2, 4, 5, 4, 5, 7, 6, 5, 4, 3, 1, 3, 2, 1, 3, 2, 3, 4, 5, 4, 3, 4, 2, 3, 2, 4, 2, 4, 2, 3, 4, 5, 4, 5, 7, 6, 5, 4, 5, 6, 5, 6, 7, 5, 6, 7, 6, 7, 8, 1, 2, 4, 3, 1, 8, 1, 2, 3, 4, 5, 4, 2, 3, 4, 3, 4, 5, 6, 7, 6, 5, 4, 5, 4, 5, 7, 8, 7, 6, 7, 5, 7, 8}
			} else if ss.runnum == 32 {
				seqid = [900]int{4, 2, 3, 4, 3, 1, 8, 6, 7, 5, 7, 8, 1, 3, 2, 1, 8, 7, 6, 5, 4, 3, 4, 3, 4, 2, 3, 1, 8, 1, 2, 1, 3, 2, 1, 2, 3, 4, 2, 3, 2, 3, 1, 8, 7, 8, 1, 3, 2, 4, 2, 3, 2, 1, 8, 7, 6, 5, 6, 5, 6, 5, 7, 6, 8, 7, 5, 4, 3, 1, 2, 4, 3, 2, 1, 8, 6, 5, 4, 2, 1, 3, 4, 5, 6, 5, 7, 6, 5, 4, 5, 7, 6, 5, 7, 5, 4, 5, 7, 6, 7, 6, 7, 5, 6, 5, 4, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 2, 3, 1, 3, 4, 3, 1, 2, 3, 4, 5, 7, 8, 7, 5, 7, 8, 6, 5, 7, 8, 7, 6, 7, 8, 6, 7, 6, 5, 7, 6, 5, 7, 5, 6, 5, 7, 6, 8, 6, 8, 6, 5, 4, 2, 3, 2, 3, 2, 3, 4, 2, 3, 1, 3, 1, 8, 1, 8, 7, 5, 6, 5, 4, 3, 4, 3, 2, 4, 3, 2, 4, 3, 1, 2, 1, 8, 7, 8, 1, 3, 2, 1, 3, 2, 4, 2, 1, 2, 1, 8, 7, 5, 6, 7, 8, 6, 7, 5, 4, 5, 4, 2, 4, 5, 7, 6, 7, 8, 7, 5, 4, 3, 1, 8, 7, 8, 7, 8, 6, 7, 8, 7, 5, 7, 5, 4, 2, 3, 1, 8, 1, 2, 3, 4, 2, 3, 2, 3, 2, 4, 2, 1, 8, 6, 5, 4, 3, 1, 2, 3, 1, 8, 6, 5, 6, 7, 6, 7, 6, 5, 4, 3, 1, 8, 1, 3, 1, 2, 4, 5, 7, 6, 8, 7, 6, 8, 6, 8, 1, 8, 6, 5, 6, 7, 5, 7, 8, 7, 8, 6, 5, 7, 6, 5, 6, 8, 1, 8, 6, 8, 6, 5, 6, 7, 8, 1, 3, 1, 8, 6, 5, 4, 3, 1, 2, 3, 1, 3, 2, 4, 3, 4, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 6, 5, 7, 6, 7, 8, 7, 6, 5, 7, 6, 5, 6, 7, 8, 6, 5, 4, 3, 1, 8, 1, 3, 1, 2, 1, 3, 2, 3, 4, 3, 2, 4, 5, 7, 6, 5, 6, 5, 6, 8, 1, 8, 7, 8, 6, 8, 7, 8, 7, 8, 1, 3, 2, 3, 4, 2, 1, 3, 4, 3, 2, 4, 2, 4, 5, 4, 3, 2, 3, 4, 3, 2, 4, 3, 1, 2, 3, 2, 4, 3, 4, 2, 1, 8, 1, 2, 3, 1, 3, 4, 3, 4, 2, 4, 2, 3, 2, 1, 8, 7, 5, 7, 6, 7, 6, 5, 4, 2, 1, 2, 3, 4, 3, 4, 3, 4, 2, 4, 3, 1, 3, 2, 3, 1, 2, 3, 1, 2, 4, 3, 4, 3, 4, 5, 6, 8, 1, 8, 6, 7, 5, 4, 2, 3, 4, 3, 1, 8, 1, 3, 4, 3, 1, 2, 4, 3, 1, 8, 6, 5, 7, 8, 6, 7, 6, 8, 7, 8, 1, 8, 6, 7, 6, 8, 1, 3, 4, 2, 3, 2, 3, 4, 5, 7, 5, 6, 8, 7, 5, 4, 3, 1, 2, 1, 2, 3, 4, 3, 1, 3, 1, 3, 1, 3, 4, 2, 1, 8, 6, 7, 8, 7, 5, 7, 5, 7, 6, 7, 5, 6, 8, 1, 2, 4, 2, 3, 2, 3, 1, 8, 1, 2, 4, 2, 1, 8, 1, 3, 2, 4, 5, 6, 8, 1, 2, 1, 3, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 4, 5, 4, 3, 2, 1, 8, 7, 8, 6, 8, 7, 5, 7, 6, 7, 6, 8, 6, 7, 5, 7, 6, 7, 6, 8, 7, 8, 7, 8, 6, 8, 6, 8, 1, 2, 4, 2, 3, 2, 3, 1, 8, 1, 3, 4, 3, 1, 2, 3, 4, 3, 4, 2, 1, 3, 1, 2, 3, 1, 3, 1, 8, 1, 8, 1, 3, 4, 5, 4, 2, 3, 2, 4, 3, 2, 1, 8, 6, 5, 4, 5, 4, 2, 1, 8, 1, 3, 2, 4, 5, 4, 2, 3, 4, 3, 1, 8, 7, 8, 1, 2, 3, 4, 3, 4, 5, 7, 6, 7, 6, 7, 8, 1, 3, 2, 1, 3, 2, 1, 8, 6, 7, 8, 7, 8, 1, 8, 1, 2, 1, 8, 6, 5, 7, 6, 7, 6, 8, 7, 6, 7, 8, 1, 8, 7, 6, 7, 8, 1, 3, 2, 3, 4, 2, 4, 3, 4, 3, 1, 8, 6, 8, 7, 5, 6, 7, 8, 7, 5, 4, 5, 7, 6, 7, 6, 7, 8, 6, 8, 6, 8, 1, 8, 7, 8, 1, 8, 7, 5, 6, 8, 6, 5, 6, 7, 5, 7, 8, 7, 6, 7, 6, 7, 6, 8, 1, 2, 4, 2, 4, 5, 4, 3, 1, 8, 6, 7, 6, 7, 6, 7, 5, 7, 6, 8, 1, 3, 4, 3, 2, 3, 2, 3, 4, 5, 4, 2, 4, 2, 4, 5, 4, 3, 1, 8, 6, 5, 7, 5, 4, 3, 1, 8, 1, 8, 6, 7, 5, 7, 8, 6, 7, 6, 5, 4, 3, 1, 3, 1, 8, 1, 8, 1, 2, 3, 1, 8, 7, 8, 1, 3, 4, 3, 2, 4, 5, 6, 5, 4, 2, 1}
			} else if ss.runnum == 33 {
				seqid = [900]int{8, 7, 5, 6, 8, 1, 2, 1, 8, 7, 8, 1, 2, 1, 3, 1, 2, 1, 3, 4, 2, 3, 1, 2, 1, 3, 2, 3, 2, 1, 3, 2, 4, 5, 6, 7, 5, 7, 5, 4, 2, 3, 1, 3, 1, 3, 4, 2, 1, 3, 1, 2, 3, 4, 2, 1, 2, 1, 2, 3, 2, 4, 2, 3, 1, 2, 4, 3, 4, 5, 7, 5, 7, 5, 4, 5, 7, 6, 5, 4, 2, 1, 2, 1, 3, 1, 3, 4, 3, 4, 3, 4, 3, 2, 3, 1, 8, 6, 8, 7, 6, 5, 7, 8, 6, 8, 1, 2, 1, 8, 7, 5, 4, 3, 2, 1, 8, 6, 7, 8, 6, 5, 7, 6, 7, 8, 1, 2, 1, 3, 1, 2, 3, 1, 3, 4, 3, 1, 2, 4, 3, 4, 3, 1, 8, 1, 3, 4, 5, 7, 5, 7, 8, 1, 2, 3, 1, 2, 3, 1, 3, 4, 5, 7, 5, 4, 5, 4, 2, 3, 1, 8, 6, 8, 6, 7, 5, 6, 7, 6, 5, 4, 5, 7, 6, 7, 8, 6, 7, 6, 7, 8, 6, 8, 6, 8, 7, 5, 7, 5, 4, 2, 3, 4, 3, 1, 2, 1, 8, 6, 5, 6, 5, 6, 8, 6, 5, 4, 5, 7, 6, 7, 6, 7, 6, 8, 7, 8, 6, 7, 8, 6, 8, 6, 5, 7, 6, 5, 7, 6, 8, 6, 8, 7, 6, 7, 5, 6, 5, 7, 6, 7, 5, 6, 5, 6, 7, 5, 7, 5, 7, 8, 6, 8, 6, 5, 6, 8, 6, 5, 6, 7, 5, 4, 3, 4, 3, 1, 8, 7, 8, 7, 8, 6, 5, 6, 8, 1, 3, 4, 2, 4, 2, 3, 1, 2, 3, 2, 4, 2, 4, 5, 7, 8, 7, 6, 7, 8, 1, 3, 2, 1, 2, 1, 8, 6, 7, 6, 5, 7, 8, 1, 2, 1, 3, 2, 1, 2, 4, 5, 6, 7, 8, 1, 2, 4, 5, 6, 5, 4, 5, 7, 6, 7, 5, 4, 5, 6, 5, 4, 2, 3, 1, 2, 3, 2, 3, 2, 4, 3, 1, 8, 7, 6, 8, 6, 8, 7, 5, 4, 5, 7, 8, 6, 5, 6, 5, 7, 6, 7, 6, 7, 5, 7, 5, 6, 8, 1, 8, 1, 8, 6, 5, 6, 7, 6, 8, 7, 5, 6, 7, 5, 7, 6, 8, 7, 6, 8, 1, 2, 3, 2, 3, 1, 2, 1, 8, 6, 8, 6, 8, 1, 3, 1, 8, 7, 6, 5, 6, 7, 8, 6, 8, 7, 8, 7, 8, 6, 7, 6, 5, 6, 8, 1, 3, 4, 5, 4, 5, 7, 6, 5, 4, 3, 4, 2, 3, 2, 4, 5, 7, 8, 7, 5, 7, 5, 4, 2, 1, 2, 3, 2, 4, 5, 7, 8, 1, 8, 1, 3, 1, 8, 6, 5, 6, 7, 6, 7, 6, 5, 7, 8, 7, 5, 4, 3, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8, 7, 6, 7, 5, 4, 2, 3, 4, 2, 1, 3, 1, 3, 4, 3, 2, 4, 2, 1, 3, 4, 2, 1, 2, 4, 5, 6, 5, 6, 5, 6, 7, 5, 4, 2, 4, 2, 4, 3, 1, 2, 3, 1, 3, 4, 2, 4, 5, 7, 8, 6, 7, 8, 1, 3, 2, 1, 2, 3, 4, 2, 1, 8, 7, 6, 7, 8, 6, 8, 7, 5, 4, 5, 4, 3, 2, 1, 3, 2, 1, 2, 3, 4, 3, 1, 8, 1, 2, 1, 3, 4, 3, 1, 8, 6, 8, 7, 5, 6, 8, 1, 3, 1, 3, 2, 4, 3, 4, 2, 3, 2, 1, 2, 1, 3, 2, 4, 5, 6, 7, 6, 8, 7, 8, 6, 8, 6, 7, 6, 7, 8, 1, 3, 2, 1, 8, 6, 7, 6, 5, 7, 6, 7, 5, 4, 5, 7, 6, 5, 4, 3, 2, 3, 1, 2, 1, 8, 6, 7, 6, 7, 5, 4, 2, 1, 2, 4, 3, 4, 3, 1, 3, 4, 2, 1, 2, 3, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 2, 4, 2, 1, 8, 6, 7, 8, 6, 7, 6, 7, 6, 8, 7, 8, 7, 6, 5, 7, 6, 8, 6, 7, 8, 7, 6, 7, 8, 6, 7, 8, 1, 3, 1, 2, 1, 8, 7, 5, 7, 8, 6, 5, 4, 2, 1, 3, 4, 2, 3, 4, 5, 6, 7, 8, 1, 3, 4, 2, 3, 1, 2, 1, 2, 1, 3, 1, 8, 7, 5, 7, 5, 4, 2, 4, 3, 2, 3, 4, 3, 2, 1, 8, 1, 3, 4, 5, 4, 5, 7, 8, 6, 7, 6, 5, 6, 8, 6, 8, 7, 8, 1, 3, 2, 1, 2, 4, 2, 4, 3, 1, 2, 4, 5, 6, 7, 5, 7, 6, 7, 6, 7, 6, 8, 7, 6, 8, 1, 2, 4, 5, 7, 8, 7, 5, 7, 5, 6, 8, 1, 2, 3, 1, 8, 1, 8, 7, 5, 7, 6, 7, 6, 8, 7, 5, 6, 5, 7, 5, 7, 8, 1, 2, 4, 2, 3, 2, 4, 3, 2, 1, 2, 1, 8, 6, 8, 1, 3, 1, 2, 4, 3, 2, 1, 8, 7, 5, 4, 2, 3, 1, 8, 6, 7, 5, 4, 3, 1, 3, 1, 3, 4, 2, 4, 5}
			} else if ss.runnum == 34 {
				seqid = [900]int{7, 6, 7, 8, 1, 3, 4, 5, 6, 7, 5, 7, 8, 1, 2, 4, 5, 4, 3, 2, 3, 4, 2, 4, 2, 4, 5, 7, 6, 7, 8, 7, 8, 7, 6, 8, 7, 8, 1, 3, 2, 4, 5, 4, 2, 3, 4, 3, 4, 3, 4, 3, 1, 8, 1, 8, 7, 8, 7, 8, 1, 3, 2, 3, 4, 3, 2, 3, 4, 2, 4, 2, 1, 3, 2, 4, 3, 4, 3, 1, 2, 4, 3, 4, 3, 1, 2, 1, 3, 2, 4, 3, 2, 1, 8, 1, 2, 3, 2, 3, 4, 2, 3, 4, 5, 4, 5, 7, 8, 7, 8, 1, 3, 4, 2, 3, 2, 3, 2, 1, 3, 2, 1, 2, 4, 2, 1, 8, 7, 8, 6, 8, 6, 5, 7, 5, 6, 8, 7, 5, 4, 2, 1, 8, 6, 5, 6, 5, 6, 7, 6, 5, 7, 5, 4, 2, 4, 2, 1, 3, 4, 5, 4, 5, 7, 8, 6, 5, 7, 5, 7, 5, 7, 8, 7, 5, 6, 5, 7, 6, 7, 5, 7, 6, 5, 4, 2, 4, 3, 2, 1, 8, 6, 8, 6, 8, 6, 7, 8, 7, 8, 1, 2, 4, 3, 2, 3, 4, 3, 4, 3, 2, 1, 8, 7, 8, 1, 3, 2, 3, 1, 2, 1, 3, 2, 4, 2, 1, 3, 4, 2, 4, 2, 1, 8, 7, 8, 1, 3, 1, 3, 2, 4, 2, 3, 4, 3, 2, 3, 1, 8, 7, 6, 8, 6, 5, 6, 7, 8, 1, 3, 1, 3, 1, 2, 4, 3, 2, 1, 8, 1, 2, 3, 1, 2, 3, 1, 2, 1, 8, 1, 2, 3, 1, 8, 6, 8, 1, 2, 4, 2, 3, 4, 5, 4, 5, 6, 8, 7, 5, 6, 8, 7, 6, 5, 7, 8, 1, 3, 2, 4, 2, 1, 8, 7, 8, 7, 5, 7, 5, 7, 5, 4, 5, 4, 5, 4, 5, 4, 2, 1, 8, 7, 5, 6, 8, 6, 7, 6, 7, 6, 5, 6, 7, 8, 6, 8, 1, 2, 4, 5, 4, 5, 4, 5, 6, 5, 7, 5, 6, 7, 6, 7, 5, 7, 5, 6, 5, 4, 3, 1, 3, 2, 1, 8, 6, 7, 8, 7, 6, 5, 4, 2, 1, 8, 6, 7, 5, 4, 5, 4, 3, 2, 3, 4, 3, 4, 2, 1, 8, 1, 8, 7, 5, 4, 3, 2, 1, 8, 1, 3, 4, 5, 6, 8, 7, 5, 4, 2, 3, 4, 3, 2, 3, 4, 5, 4, 5, 7, 6, 8, 1, 8, 7, 5, 4, 2, 4, 5, 6, 8, 6, 5, 4, 3, 4, 2, 3, 1, 3, 1, 3, 2, 4, 5, 7, 6, 5, 6, 5, 6, 7, 8, 1, 8, 7, 8, 7, 6, 7, 5, 6, 8, 6, 7, 6, 7, 8, 6, 5, 4, 3, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 8, 1, 2, 1, 8, 1, 3, 4, 2, 3, 4, 2, 1, 3, 1, 3, 2, 1, 3, 1, 2, 3, 1, 8, 7, 6, 7, 6, 7, 5, 7, 8, 1, 8, 7, 5, 4, 2, 4, 3, 4, 3, 2, 3, 4, 2, 3, 4, 5, 6, 5, 4, 5, 7, 6, 8, 7, 5, 4, 2, 4, 3, 4, 3, 2, 3, 4, 2, 3, 4, 2, 4, 2, 4, 2, 1, 3, 4, 5, 4, 3, 4, 5, 4, 3, 2, 4, 5, 7, 5, 4, 2, 4, 3, 1, 3, 2, 3, 4, 3, 2, 1, 3, 4, 5, 4, 3, 4, 2, 3, 4, 2, 1, 2, 4, 2, 1, 3, 1, 8, 6, 7, 6, 5, 7, 6, 5, 7, 5, 6, 5, 6, 8, 1, 8, 1, 2, 1, 8, 7, 8, 7, 8, 1, 3, 1, 3, 2, 4, 2, 1, 3, 1, 8, 7, 6, 8, 7, 5, 7, 5, 7, 8, 7, 6, 5, 4, 2, 1, 8, 1, 2, 4, 2, 1, 3, 1, 8, 1, 8, 7, 6, 5, 7, 5, 6, 7, 6, 8, 7, 6, 8, 6, 5, 6, 8, 1, 8, 7, 5, 6, 8, 6, 5, 6, 8, 6, 5, 6, 7, 8, 7, 5, 6, 5, 4, 3, 1, 8, 7, 8, 7, 6, 7, 6, 5, 6, 5, 7, 8, 7, 8, 1, 3, 2, 1, 3, 4, 2, 4, 5, 4, 3, 2, 4, 3, 4, 2, 1, 2, 1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 2, 4, 5, 6, 7, 5, 4, 3, 2, 1, 2, 4, 3, 2, 4, 2, 3, 4, 3, 4, 2, 3, 4, 5, 4, 2, 1, 2, 1, 2, 4, 3, 4, 5, 6, 8, 6, 7, 6, 5, 4, 3, 1, 3, 2, 1, 2, 3, 2, 4, 5, 6, 7, 8, 6, 7, 6, 8, 6, 5, 7, 5, 7, 8, 6, 5, 6, 5, 6, 8, 6, 8, 1, 2, 4, 3, 1, 8, 6, 8, 7, 6, 5, 7, 5, 4, 2, 3, 2, 4, 3, 1, 3, 2, 3, 1, 3, 2, 4, 3, 1, 2, 3, 4, 3, 2, 4, 2, 1, 2, 4, 3, 4, 5, 7, 8, 1, 3, 2, 3, 4, 5, 7, 6, 7, 8, 7, 5, 6, 7, 6, 5, 7, 5, 7, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6}
			} else if ss.runnum == 35 {
				seqid = [900]int{7, 8, 6, 5, 7, 5, 6, 7, 8, 1, 2, 4, 3, 2, 1, 3, 1, 3, 1, 2, 1, 2, 3, 2, 4, 2, 3, 4, 5, 7, 5, 4, 5, 6, 8, 7, 8, 7, 5, 6, 8, 1, 3, 2, 4, 2, 3, 1, 8, 7, 8, 1, 8, 7, 6, 7, 6, 7, 6, 7, 6, 7, 5, 6, 7, 6, 8, 1, 2, 1, 2, 3, 1, 3, 1, 8, 1, 8, 6, 5, 7, 6, 5, 6, 5, 4, 5, 4, 2, 3, 2, 3, 1, 2, 4, 5, 4, 2, 3, 2, 4, 3, 2, 1, 3, 2, 4, 2, 4, 5, 7, 6, 5, 6, 5, 4, 3, 1, 2, 4, 3, 4, 5, 4, 5, 4, 2, 4, 5, 7, 8, 1, 3, 2, 3, 4, 3, 2, 4, 5, 6, 5, 6, 8, 6, 8, 7, 8, 6, 5, 4, 2, 1, 8, 6, 8, 7, 6, 7, 8, 7, 8, 1, 3, 1, 8, 6, 5, 4, 5, 4, 2, 4, 2, 3, 2, 4, 3, 4, 2, 3, 2, 3, 2, 1, 8, 1, 3, 1, 8, 6, 8, 7, 6, 7, 5, 7, 5, 4, 5, 6, 8, 1, 2, 4, 5, 6, 8, 6, 7, 6, 8, 7, 6, 8, 6, 5, 6, 5, 6, 5, 6, 8, 6, 5, 7, 8, 6, 5, 4, 3, 2, 4, 3, 2, 4, 5, 7, 6, 8, 1, 2, 4, 3, 4, 3, 2, 3, 2, 3, 4, 2, 4, 5, 7, 8, 1, 8, 1, 8, 6, 7, 5, 4, 5, 7, 8, 1, 3, 2, 3, 2, 3, 2, 4, 2, 4, 5, 6, 7, 5, 7, 5, 4, 2, 4, 5, 6, 8, 7, 5, 6, 8, 6, 8, 7, 5, 7, 5, 7, 5, 6, 7, 8, 1, 2, 3, 2, 3, 2, 4, 5, 4, 2, 4, 3, 1, 2, 4, 2, 1, 2, 1, 8, 7, 5, 4, 3, 1, 2, 1, 3, 1, 3, 4, 2, 1, 8, 7, 6, 5, 6, 5, 7, 6, 8, 7, 5, 7, 6, 7, 5, 4, 3, 2, 4, 2, 4, 3, 1, 2, 4, 3, 1, 2, 3, 4, 2, 3, 2, 4, 2, 4, 3, 4, 3, 1, 2, 4, 2, 1, 3, 4, 3, 1, 8, 7, 5, 4, 3, 2, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 8, 6, 8, 6, 8, 6, 5, 7, 5, 4, 2, 4, 5, 7, 6, 7, 6, 8, 7, 6, 5, 7, 6, 5, 6, 7, 8, 7, 5, 7, 8, 1, 8, 1, 2, 3, 2, 1, 8, 1, 8, 1, 3, 4, 3, 1, 8, 6, 7, 5, 7, 8, 1, 8, 6, 7, 6, 8, 1, 8, 6, 8, 1, 2, 3, 4, 2, 3, 1, 8, 1, 2, 3, 1, 8, 6, 7, 5, 7, 5, 7, 5, 6, 5, 7, 5, 6, 8, 1, 8, 6, 7, 8, 6, 5, 6, 5, 4, 5, 6, 8, 1, 3, 2, 3, 4, 2, 4, 3, 2, 3, 4, 5, 7, 6, 8, 1, 3, 1, 2, 1, 3, 4, 2, 3, 4, 2, 4, 3, 4, 3, 2, 3, 4, 5, 6, 8, 1, 8, 1, 3, 1, 8, 6, 7, 8, 7, 6, 5, 4, 5, 7, 5, 4, 3, 4, 3, 4, 3, 4, 2, 1, 2, 3, 1, 3, 2, 4, 2, 4, 5, 6, 8, 7, 8, 6, 7, 8, 7, 5, 6, 8, 1, 8, 6, 7, 6, 8, 1, 2, 4, 3, 1, 3, 1, 8, 6, 8, 6, 7, 8, 7, 8, 1, 8, 1, 3, 4, 2, 4, 5, 7, 5, 6, 7, 5, 4, 5, 7, 8, 1, 2, 4, 2, 1, 3, 4, 2, 1, 3, 1, 8, 7, 5, 4, 5, 4, 5, 7, 8, 7, 8, 1, 2, 1, 3, 4, 5, 7, 6, 8, 1, 3, 2, 1, 2, 4, 5, 7, 8, 1, 3, 4, 3, 4, 2, 1, 3, 1, 3, 2, 4, 2, 4, 5, 6, 5, 6, 7, 5, 7, 8, 7, 5, 6, 7, 6, 7, 5, 6, 5, 4, 5, 4, 2, 1, 8, 1, 2, 1, 8, 1, 2, 3, 1, 8, 6, 7, 8, 1, 8, 1, 8, 1, 3, 4, 5, 6, 5, 6, 8, 1, 8, 1, 8, 7, 6, 5, 6, 8, 6, 7, 5, 6, 8, 7, 5, 7, 6, 7, 8, 6, 5, 4, 3, 4, 2, 3, 1, 2, 3, 4, 5, 6, 5, 6, 5, 4, 3, 1, 2, 1, 2, 1, 2, 4, 5, 6, 5, 7, 6, 5, 4, 3, 1, 8, 6, 5, 7, 8, 6, 5, 4, 3, 4, 5, 7, 6, 7, 5, 6, 7, 5, 7, 8, 6, 8, 7, 8, 1, 3, 2, 4, 3, 4, 2, 1, 3, 4, 3, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 3, 2, 1, 3, 1, 2, 1, 3, 4, 2, 4, 2, 4, 2, 1, 8, 6, 5, 7, 8, 7, 5, 6, 7, 6, 8, 6, 5, 6, 5, 6, 7, 8, 7, 5, 6, 7, 8, 7, 5, 6, 7, 5, 6, 8, 1, 2, 3, 1, 8, 1, 3, 4, 5, 4, 2, 3, 4, 5, 7, 6, 5, 4, 3, 4, 5, 6, 8, 7, 5, 4, 2, 1, 2}
			} else if ss.runnum == 36 {
				seqid = [900]int{2, 1, 8, 6, 8, 1, 8, 1, 2, 4, 2, 4, 2, 1, 3, 1, 3, 4, 5, 7, 8, 7, 8, 1, 3, 4, 5, 7, 6, 5, 6, 7, 6, 7, 8, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 6, 7, 8, 7, 6, 8, 6, 5, 4, 3, 4, 5, 4, 3, 1, 3, 2, 1, 2, 1, 3, 2, 1, 3, 1, 3, 4, 2, 1, 8, 1, 2, 1, 8, 1, 2, 1, 3, 2, 1, 3, 1, 3, 1, 2, 3, 4, 3, 4, 3, 4, 2, 4, 5, 7, 5, 7, 5, 6, 5, 4, 3, 2, 4, 3, 1, 8, 6, 5, 7, 5, 7, 5, 7, 8, 6, 5, 7, 5, 4, 2, 1, 8, 1, 2, 1, 2, 3, 4, 2, 4, 5, 6, 7, 8, 6, 5, 4, 5, 4, 5, 6, 7, 6, 8, 6, 7, 8, 1, 3, 1, 3, 4, 3, 2, 1, 2, 4, 5, 7, 5, 4, 3, 2, 1, 2, 3, 2, 3, 1, 2, 1, 8, 1, 8, 6, 7, 6, 8, 7, 8, 6, 7, 6, 7, 5, 6, 5, 6, 5, 6, 8, 7, 6, 7, 5, 7, 6, 7, 8, 6, 7, 8, 6, 5, 4, 3, 1, 8, 7, 6, 5, 4, 2, 1, 2, 4, 2, 1, 2, 1, 3, 1, 8, 1, 8, 6, 8, 1, 8, 7, 8, 6, 7, 5, 4, 5, 6, 8, 7, 8, 7, 6, 7, 5, 4, 3, 1, 3, 4, 2, 3, 1, 3, 2, 3, 4, 5, 6, 5, 7, 6, 5, 4, 5, 7, 6, 5, 7, 6, 8, 1, 3, 4, 3, 2, 4, 5, 6, 7, 6, 8, 1, 2, 3, 4, 5, 4, 5, 4, 5, 7, 8, 6, 7, 8, 1, 8, 6, 8, 6, 7, 8, 6, 8, 1, 2, 3, 1, 8, 7, 8, 6, 5, 7, 5, 7, 8, 7, 8, 6, 7, 8, 6, 5, 6, 5, 7, 8, 7, 6, 8, 1, 3, 4, 2, 3, 1, 8, 6, 8, 7, 8, 7, 8, 1, 2, 1, 3, 2, 1, 3, 1, 2, 4, 3, 2, 3, 4, 2, 4, 5, 6, 5, 4, 3, 4, 5, 7, 6, 8, 1, 2, 4, 2, 1, 3, 2, 3, 1, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 2, 1, 3, 2, 3, 2, 1, 3, 4, 3, 1, 2, 1, 8, 7, 8, 1, 3, 4, 5, 6, 8, 7, 5, 4, 3, 4, 5, 6, 7, 6, 8, 1, 8, 6, 7, 8, 1, 3, 2, 3, 2, 1, 2, 3, 2, 4, 3, 1, 3, 1, 8, 1, 8, 6, 5, 4, 2, 1, 3, 4, 2, 1, 2, 1, 8, 6, 5, 4, 2, 3, 1, 2, 1, 2, 3, 1, 8, 6, 8, 1, 3, 4, 5, 4, 5, 4, 2, 3, 1, 2, 3, 2, 1, 8, 7, 6, 8, 7, 6, 7, 8, 7, 8, 6, 8, 6, 7, 8, 7, 8, 6, 8, 7, 8, 6, 8, 6, 8, 1, 3, 1, 3, 4, 3, 4, 3, 1, 2, 1, 2, 3, 2, 3, 4, 3, 1, 2, 3, 2, 4, 2, 1, 3, 4, 3, 4, 5, 4, 2, 3, 4, 5, 6, 7, 8, 6, 7, 5, 6, 8, 1, 8, 7, 8, 7, 8, 6, 5, 7, 5, 7, 5, 4, 2, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 3, 1, 3, 2, 1, 8, 7, 8, 7, 6, 7, 8, 1, 2, 3, 2, 4, 3, 4, 3, 1, 3, 1, 8, 1, 8, 1, 8, 6, 7, 8, 6, 8, 6, 8, 6, 7, 5, 6, 7, 5, 7, 8, 6, 5, 4, 5, 6, 7, 8, 1, 8, 6, 5, 7, 5, 4, 5, 4, 3, 2, 1, 3, 2, 4, 5, 4, 5, 7, 5, 6, 7, 8, 6, 8, 1, 2, 3, 2, 1, 2, 1, 8, 7, 8, 1, 3, 1, 2, 1, 3, 4, 3, 1, 3, 4, 5, 6, 7, 5, 7, 5, 6, 5, 7, 6, 7, 5, 7, 8, 6, 5, 6, 7, 6, 7, 5, 6, 7, 8, 1, 2, 3, 1, 2, 1, 3, 4, 2, 4, 2, 1, 3, 1, 3, 4, 5, 4, 3, 2, 1, 8, 7, 5, 7, 8, 7, 8, 1, 3, 1, 3, 4, 2, 4, 5, 7, 5, 4, 3, 1, 2, 1, 2, 4, 2, 4, 2, 1, 8, 6, 5, 4, 2, 1, 3, 1, 8, 1, 2, 4, 2, 1, 2, 4, 5, 4, 2, 4, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 2, 3, 1, 2, 3, 1, 8, 6, 8, 1, 2, 1, 2, 3, 1, 2, 1, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 8, 7, 6, 8, 6, 5, 6, 7, 5, 7, 6, 8, 1, 8, 7, 8, 7, 5, 7, 8, 6, 8, 1, 2, 3, 1, 2, 1, 8, 7, 6, 8, 1, 8, 1, 8, 1, 8, 1, 3, 4, 5, 7, 8, 1, 2, 3, 4, 2, 4, 5, 6, 8, 7, 5, 4, 5, 6, 7, 6, 7, 6, 7, 8, 6, 7, 5, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 5, 7, 6, 5, 4, 5, 4, 5, 6, 5, 4, 5, 7, 6, 8, 7}
			} else if ss.runnum == 37 {
				seqid = [900]int{1, 3, 4, 2, 3, 1, 3, 1, 2, 4, 3, 2, 4, 3, 1, 2, 1, 2, 4, 5, 7, 5, 4, 3, 2, 1, 3, 4, 3, 4, 5, 6, 5, 7, 8, 1, 2, 4, 3, 1, 3, 4, 3, 4, 5, 4, 2, 1, 8, 1, 8, 7, 8, 7, 5, 4, 2, 3, 4, 2, 3, 2, 4, 5, 6, 5, 6, 8, 7, 6, 8, 6, 8, 7, 8, 6, 5, 6, 7, 5, 7, 6, 5, 6, 7, 5, 4, 3, 1, 8, 7, 8, 7, 6, 8, 1, 3, 2, 1, 8, 1, 8, 7, 8, 6, 8, 6, 7, 8, 6, 7, 5, 7, 8, 7, 6, 5, 7, 8, 1, 3, 1, 2, 4, 2, 1, 2, 4, 3, 4, 5, 6, 5, 7, 5, 4, 5, 6, 8, 6, 5, 4, 3, 4, 2, 3, 1, 8, 7, 5, 7, 6, 5, 6, 5, 6, 5, 7, 5, 4, 2, 4, 5, 6, 7, 8, 6, 8, 7, 6, 5, 4, 2, 3, 4, 5, 7, 5, 7, 5, 7, 8, 6, 7, 5, 7, 6, 5, 4, 2, 3, 1, 2, 3, 4, 3, 4, 3, 2, 3, 1, 3, 1, 8, 7, 8, 7, 6, 7, 5, 4, 5, 6, 7, 6, 7, 6, 5, 4, 5, 4, 2, 3, 2, 4, 5, 7, 6, 7, 6, 5, 7, 8, 7, 6, 5, 6, 7, 6, 8, 1, 3, 2, 4, 2, 1, 3, 2, 3, 2, 4, 5, 7, 6, 8, 7, 6, 8, 1, 2, 1, 8, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 3, 4, 2, 1, 3, 4, 5, 4, 2, 3, 1, 8, 1, 2, 1, 3, 1, 8, 6, 8, 7, 5, 6, 8, 1, 8, 6, 8, 1, 2, 3, 4, 3, 2, 1, 3, 1, 8, 7, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 1, 2, 3, 2, 1, 8, 6, 8, 7, 8, 1, 8, 6, 7, 8, 6, 5, 4, 5, 6, 7, 8, 6, 7, 6, 8, 7, 8, 1, 3, 2, 1, 8, 1, 3, 2, 3, 2, 3, 2, 1, 2, 4, 3, 1, 8, 7, 6, 5, 7, 5, 6, 5, 6, 7, 5, 4, 2, 4, 3, 1, 3, 2, 3, 2, 1, 8, 1, 2, 4, 3, 2, 4, 5, 7, 5, 6, 8, 1, 3, 1, 2, 3, 2, 4, 3, 1, 2, 4, 2, 3, 1, 3, 2, 3, 1, 8, 6, 7, 8, 1, 8, 1, 2, 1, 3, 1, 3, 2, 4, 2, 1, 2, 4, 2, 4, 2, 3, 1, 2, 1, 8, 6, 7, 5, 4, 3, 1, 8, 1, 2, 1, 3, 2, 3, 4, 5, 7, 6, 5, 4, 2, 1, 2, 4, 5, 7, 6, 5, 7, 6, 8, 7, 6, 7, 8, 6, 5, 6, 7, 6, 8, 6, 5, 6, 8, 6, 5, 4, 3, 1, 2, 1, 8, 1, 3, 4, 3, 1, 2, 1, 8, 7, 6, 5, 7, 6, 8, 6, 7, 8, 7, 5, 6, 5, 4, 3, 2, 4, 3, 1, 2, 1, 3, 1, 2, 4, 3, 2, 3, 2, 1, 8, 7, 8, 7, 5, 6, 8, 6, 8, 7, 6, 7, 8, 1, 2, 1, 8, 7, 6, 5, 6, 7, 5, 7, 8, 6, 7, 8, 6, 7, 5, 7, 6, 8, 1, 8, 7, 5, 4, 5, 4, 2, 4, 2, 3, 1, 2, 4, 2, 3, 2, 1, 8, 7, 5, 6, 5, 7, 8, 1, 8, 1, 8, 6, 7, 5, 6, 7, 5, 7, 8, 1, 3, 1, 8, 6, 7, 5, 7, 5, 4, 2, 1, 8, 7, 5, 7, 6, 5, 4, 2, 4, 2, 3, 4, 2, 1, 3, 2, 3, 4, 3, 1, 3, 1, 3, 2, 4, 2, 4, 3, 1, 8, 1, 2, 3, 2, 1, 3, 4, 5, 6, 8, 7, 6, 5, 4, 2, 1, 2, 3, 4, 2, 1, 2, 4, 5, 4, 5, 6, 8, 1, 3, 2, 3, 2, 3, 4, 5, 7, 8, 6, 8, 1, 2, 1, 3, 1, 2, 1, 8, 6, 5, 6, 5, 4, 2, 1, 8, 6, 8, 1, 3, 2, 3, 2, 1, 2, 4, 5, 6, 7, 5, 6, 8, 1, 2, 1, 2, 3, 4, 2, 4, 3, 2, 3, 4, 5, 6, 8, 1, 2, 3, 1, 3, 2, 3, 2, 4, 2, 3, 4, 5, 6, 7, 8, 1, 3, 1, 8, 1, 8, 1, 8, 7, 6, 8, 7, 8, 7, 6, 7, 5, 6, 5, 4, 3, 4, 3, 1, 2, 1, 8, 7, 6, 7, 6, 5, 7, 8, 6, 5, 7, 5, 4, 5, 7, 5, 7, 6, 7, 6, 7, 6, 5, 7, 5, 4, 5, 6, 8, 7, 8, 6, 7, 6, 7, 8, 6, 8, 1, 3, 2, 1, 3, 1, 3, 4, 3, 4, 5, 6, 5, 6, 7, 5, 4, 3, 4, 3, 4, 2, 1, 3, 1, 8, 7, 5, 6, 7, 8, 1, 2, 4, 2, 3, 2, 1, 2, 1, 8, 1, 8, 7, 8, 1, 8, 7, 5, 4, 2, 3, 2, 4, 5, 4, 2, 4, 2, 4, 3, 4, 3, 4, 5, 6, 5, 6, 7, 5, 6, 8, 1, 2, 1, 3, 1, 2, 1, 3, 4, 5, 6, 5, 6}
			} else if ss.runnum == 38 {
				seqid = [900]int{6, 5, 6, 5, 6, 8, 7, 8, 6, 5, 7, 8, 7, 8, 7, 8, 6, 8, 6, 7, 5, 6, 7, 8, 7, 6, 5, 6, 5, 6, 8, 7, 6, 5, 4, 3, 4, 2, 3, 2, 1, 8, 1, 3, 1, 3, 1, 2, 4, 3, 4, 3, 4, 2, 1, 2, 1, 8, 7, 6, 8, 1, 8, 1, 2, 4, 5, 4, 2, 4, 2, 1, 8, 6, 5, 6, 8, 6, 8, 6, 5, 4, 5, 4, 3, 2, 3, 1, 8, 1, 3, 1, 8, 1, 3, 2, 1, 8, 6, 8, 6, 8, 7, 5, 6, 8, 7, 5, 4, 5, 6, 7, 8, 1, 3, 2, 3, 1, 8, 6, 8, 7, 8, 7, 8, 7, 8, 6, 5, 6, 7, 6, 5, 7, 5, 6, 8, 6, 8, 1, 2, 3, 1, 2, 4, 5, 7, 6, 8, 6, 5, 7, 5, 4, 2, 1, 2, 4, 2, 1, 8, 6, 8, 1, 8, 6, 7, 6, 8, 1, 8, 6, 5, 4, 5, 7, 6, 5, 4, 3, 1, 8, 6, 5, 4, 5, 4, 2, 3, 4, 5, 4, 2, 1, 2, 1, 2, 3, 2, 1, 8, 6, 7, 5, 6, 5, 4, 3, 4, 5, 7, 8, 7, 5, 4, 2, 4, 3, 4, 5, 7, 5, 4, 5, 7, 8, 1, 2, 3, 4, 3, 4, 3, 2, 4, 3, 2, 1, 2, 3, 4, 5, 6, 8, 6, 5, 4, 2, 1, 2, 3, 1, 8, 7, 5, 7, 5, 4, 5, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 3, 1, 2, 3, 1, 8, 6, 5, 4, 5, 6, 8, 1, 8, 1, 8, 6, 7, 5, 4, 5, 7, 5, 4, 5, 4, 3, 1, 2, 4, 3, 1, 2, 4, 5, 6, 7, 6, 5, 6, 5, 7, 5, 4, 5, 6, 5, 4, 3, 2, 3, 4, 2, 3, 2, 4, 5, 7, 6, 8, 7, 6, 8, 7, 6, 7, 8, 6, 7, 5, 4, 2, 3, 4, 3, 1, 2, 1, 2, 4, 5, 4, 3, 4, 3, 1, 8, 1, 3, 2, 3, 2, 4, 2, 1, 8, 6, 5, 6, 5, 4, 2, 4, 2, 4, 3, 2, 3, 2, 1, 3, 1, 3, 4, 5, 4, 3, 1, 8, 7, 6, 7, 5, 6, 8, 1, 2, 1, 8, 1, 3, 2, 1, 8, 6, 7, 5, 7, 8, 7, 8, 1, 3, 2, 1, 3, 4, 2, 4, 3, 4, 3, 2, 4, 5, 4, 2, 3, 2, 4, 2, 3, 2, 1, 8, 7, 8, 6, 8, 7, 6, 5, 4, 3, 2, 1, 3, 4, 5, 7, 6, 5, 6, 5, 7, 8, 6, 5, 7, 5, 7, 5, 4, 3, 1, 8, 6, 5, 4, 2, 4, 2, 1, 3, 4, 2, 4, 3, 1, 8, 1, 3, 4, 3, 1, 3, 4, 3, 2, 3, 2, 4, 5, 4, 5, 7, 5, 7, 6, 7, 6, 5, 7, 6, 8, 1, 3, 2, 1, 3, 4, 2, 4, 3, 2, 4, 5, 4, 5, 7, 8, 7, 5, 7, 5, 7, 8, 7, 8, 6, 8, 6, 7, 6, 8, 1, 3, 4, 2, 1, 2, 1, 3, 4, 3, 2, 1, 3, 2, 1, 2, 1, 8, 7, 8, 6, 7, 5, 4, 2, 3, 4, 3, 1, 8, 7, 6, 8, 1, 8, 6, 5, 4, 3, 4, 3, 2, 1, 3, 2, 4, 3, 2, 4, 5, 4, 2, 1, 2, 4, 2, 1, 2, 4, 3, 1, 2, 1, 8, 7, 5, 7, 5, 4, 2, 3, 4, 3, 2, 4, 5, 7, 8, 6, 5, 4, 5, 4, 3, 1, 2, 4, 2, 4, 3, 4, 5, 4, 3, 2, 4, 5, 7, 5, 6, 8, 7, 6, 5, 6, 8, 1, 3, 4, 5, 6, 8, 7, 6, 5, 4, 3, 1, 2, 3, 2, 3, 2, 4, 3, 1, 2, 1, 3, 4, 3, 2, 1, 2, 3, 1, 2, 4, 3, 2, 3, 1, 8, 7, 6, 8, 1, 8, 7, 5, 4, 3, 2, 3, 2, 1, 2, 1, 8, 6, 7, 6, 7, 5, 7, 8, 1, 2, 4, 5, 6, 7, 6, 8, 1, 3, 4, 3, 2, 4, 5, 7, 5, 7, 6, 5, 6, 5, 6, 7, 8, 6, 5, 4, 3, 1, 2, 1, 8, 6, 8, 7, 6, 5, 6, 8, 1, 3, 1, 8, 7, 6, 8, 1, 3, 2, 3, 2, 3, 1, 8, 1, 2, 4, 5, 7, 8, 6, 5, 7, 5, 4, 3, 2, 3, 2, 1, 2, 1, 2, 3, 2, 4, 3, 1, 2, 4, 2, 1, 2, 3, 4, 5, 7, 6, 8, 1, 3, 4, 2, 3, 4, 5, 7, 8, 1, 3, 4, 3, 2, 3, 2, 4, 3, 1, 8, 1, 3, 2, 4, 5, 4, 2, 4, 3, 1, 3, 2, 1, 3, 1, 3, 4, 5, 4, 5, 4, 5, 7, 8, 6, 8, 6, 5, 6, 8, 1, 2, 3, 4, 3, 2, 4, 2, 4, 5, 7, 8, 6, 7, 6, 5, 4, 2, 1, 8, 7, 6, 7, 8, 6, 5, 4, 3, 1, 8, 7, 5, 4, 3, 4, 3, 4, 5, 4, 2, 4, 2, 3, 1, 8, 7, 5, 7, 6, 7, 6, 7, 5, 6, 5, 7, 6, 5, 4, 5}
			} else if ss.runnum == 39 {
				seqid = [900]int{7, 6, 5, 6, 5, 4, 2, 1, 2, 4, 2, 3, 4, 2, 4, 5, 4, 3, 1, 3, 1, 8, 6, 5, 7, 5, 7, 8, 1, 3, 4, 3, 1, 8, 1, 8, 1, 3, 4, 3, 4, 5, 7, 5, 7, 6, 5, 4, 2, 3, 4, 2, 3, 4, 2, 3, 1, 2, 4, 3, 4, 3, 2, 1, 8, 6, 5, 4, 5, 7, 5, 6, 7, 5, 4, 2, 4, 3, 4, 5, 7, 8, 1, 3, 1, 2, 4, 2, 3, 4, 2, 3, 2, 1, 2, 4, 3, 2, 4, 2, 1, 3, 2, 4, 2, 1, 3, 1, 8, 1, 2, 3, 1, 8, 1, 3, 1, 8, 7, 5, 7, 5, 7, 8, 7, 5, 6, 8, 1, 8, 1, 8, 6, 5, 6, 8, 6, 7, 6, 7, 5, 4, 3, 4, 3, 4, 5, 6, 5, 4, 5, 7, 5, 6, 5, 4, 2, 1, 3, 1, 2, 4, 3, 4, 2, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 4, 5, 4, 5, 6, 8, 1, 3, 4, 3, 4, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1, 3, 2, 1, 8, 6, 8, 1, 3, 1, 2, 4, 2, 1, 3, 1, 3, 4, 3, 2, 4, 2, 1, 3, 2, 4, 2, 3, 4, 5, 6, 5, 7, 8, 6, 8, 6, 8, 7, 6, 7, 8, 7, 6, 5, 4, 2, 4, 2, 4, 3, 4, 5, 7, 8, 1, 2, 1, 2, 4, 5, 4, 3, 2, 1, 2, 3, 4, 3, 1, 8, 1, 8, 1, 2, 4, 5, 7, 5, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 5, 7, 8, 1, 3, 1, 3, 4, 2, 1, 3, 2, 3, 1, 2, 4, 5, 4, 2, 4, 5, 6, 8, 6, 5, 7, 5, 6, 5, 7, 8, 1, 2, 4, 2, 1, 3, 1, 2, 1, 3, 1, 3, 2, 4, 3, 2, 3, 1, 8, 1, 3, 2, 1, 3, 2, 3, 1, 2, 1, 3, 4, 2, 3, 1, 8, 7, 6, 7, 5, 6, 8, 6, 5, 4, 5, 6, 5, 7, 8, 1, 3, 1, 3, 1, 3, 4, 3, 1, 2, 4, 3, 4, 2, 4, 5, 4, 5, 7, 6, 7, 6, 8, 7, 5, 7, 8, 1, 8, 6, 7, 6, 8, 6, 5, 7, 8, 6, 5, 6, 5, 4, 2, 1, 2, 4, 2, 1, 8, 1, 2, 3, 4, 3, 1, 2, 4, 5, 7, 6, 7, 8, 6, 5, 7, 5, 4, 3, 1, 8, 1, 2, 3, 4, 5, 4, 5, 7, 5, 4, 2, 1, 3, 2, 1, 2, 4, 5, 7, 8, 1, 2, 3, 4, 3, 1, 2, 3, 1, 2, 3, 1, 8, 6, 7, 8, 6, 7, 5, 4, 2, 4, 2, 3, 1, 8, 6, 8, 1, 8, 6, 5, 4, 5, 7, 8, 7, 6, 8, 7, 6, 7, 8, 6, 5, 7, 5, 6, 8, 6, 5, 4, 2, 1, 8, 1, 2, 4, 5, 7, 6, 7, 8, 7, 6, 8, 7, 8, 1, 8, 6, 5, 6, 5, 6, 7, 5, 7, 8, 7, 8, 1, 3, 2, 4, 2, 3, 2, 3, 1, 2, 3, 4, 2, 1, 3, 1, 2, 4, 5, 7, 8, 6, 5, 4, 3, 2, 1, 2, 4, 2, 1, 2, 4, 3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 3, 2, 4, 5, 7, 5, 4, 5, 6, 5, 4, 2, 1, 3, 1, 8, 7, 6, 5, 6, 7, 8, 1, 3, 2, 3, 4, 3, 1, 3, 2, 4, 3, 4, 3, 4, 5, 7, 6, 5, 6, 8, 7, 6, 8, 6, 7, 5, 4, 2, 3, 1, 2, 3, 1, 2, 3, 2, 4, 5, 6, 5, 6, 7, 5, 6, 7, 8, 7, 6, 5, 4, 2, 1, 8, 1, 3, 4, 2, 1, 2, 1, 2, 3, 4, 3, 4, 5, 7, 8, 7, 6, 7, 8, 6, 5, 4, 5, 4, 5, 7, 8, 1, 8, 6, 8, 7, 5, 4, 5, 7, 5, 6, 5, 7, 8, 7, 6, 8, 1, 3, 2, 1, 2, 4, 5, 6, 5, 7, 5, 4, 2, 3, 4, 2, 4, 5, 4, 2, 1, 3, 2, 4, 2, 4, 5, 7, 8, 7, 6, 7, 6, 5, 6, 5, 7, 6, 5, 4, 2, 4, 5, 4, 3, 4, 5, 7, 6, 8, 6, 7, 6, 8, 6, 8, 7, 8, 7, 8, 1, 8, 1, 3, 1, 3, 1, 3, 4, 5, 4, 5, 7, 8, 7, 6, 7, 6, 5, 6, 7, 5, 6, 8, 1, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 2, 1, 3, 4, 5, 7, 8, 7, 6, 7, 5, 4, 3, 1, 8, 1, 2, 4, 5, 7, 6, 7, 5, 6, 7, 5, 7, 6, 8, 1, 8, 1, 8, 7, 8, 6, 8, 7, 6, 5, 7, 8, 6, 5, 6, 7, 6, 7, 8, 6, 8, 1, 2, 3, 2, 4, 5, 4, 3, 4, 3, 1, 8, 6, 8, 7, 8, 1, 8, 1, 3, 4, 2, 1, 2, 4, 3, 2, 4, 5, 6, 7, 5, 4, 5, 4, 5, 4, 3, 4, 2, 4, 3, 1, 3, 2, 3, 4, 3, 4}
			}
		} else if ss.do_sequences == 1 {
			//sequences
			if ss.runnum == 0 {
				seqid = [900]int{6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1}
			} else if ss.runnum == 1 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 2 {
				seqid = [900]int{3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6}
			} else if ss.runnum == 3 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 4 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 5 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 6 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 7 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 8 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 9 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 10 {
				seqid = [900]int{3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6}
			} else if ss.runnum == 11 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 12 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 13 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 14 {
				seqid = [900]int{7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2}
			} else if ss.runnum == 15 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 16 {
				seqid = [900]int{8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3}
			} else if ss.runnum == 17 {
				seqid = [900]int{3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6}
			} else if ss.runnum == 18 {
				seqid = [900]int{8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3}
			} else if ss.runnum == 19 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 20 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 21 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 22 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 23 {
				seqid = [900]int{8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3}
			} else if ss.runnum == 24 {
				seqid = [900]int{6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1}
			} else if ss.runnum == 25 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 26 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 27 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 28 {
				seqid = [900]int{5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}
			} else if ss.runnum == 29 {
				seqid = [900]int{7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2}
			} else if ss.runnum == 30 {
				seqid = [900]int{6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1}
			} else if ss.runnum == 31 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 32 {
				seqid = [900]int{4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7}
			} else if ss.runnum == 33 {
				seqid = [900]int{8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3}
			} else if ss.runnum == 34 {
				seqid = [900]int{7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2}
			} else if ss.runnum == 35 {
				seqid = [900]int{7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2}
			} else if ss.runnum == 36 {
				seqid = [900]int{2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5}
			} else if ss.runnum == 37 {
				seqid = [900]int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4}
			} else if ss.runnum == 38 {
				seqid = [900]int{6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1}
			} else if ss.runnum == 39 {
				seqid = [900]int{7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2}
			}
		}
		fillseq := []string{}
		fillt := []string{}
		patgen.AddVocabEmpty(ss.PoolVocab, "emptyT", ntrans, plY, plX)
		if seqver == 2 {
			for i := 0; i < ss.wpvc*2; i++ { //all pools
				fillseq = []string{}
				for ii := 0; ii < npats-1; ii++ { // all trials //len(seqid)-1
					//training
					targitem:=i+0
					if ss.do_sequences == 3 { //make impossible so this is all just drift
						targitem=-1
					}
					if seqid[ii]-1 == targitem {
						if ss.seq_dcurr == 0 {
							fillseq = append(fillseq, "a") //current item
						} else {
							fillseq = append(fillseq, "smalla") 
						}
					} else if seqid[ii+1]-1 == i && ss.seq_numact == 2 {
						fillseq = append(fillseq, "a") //next item
					} else {
						gbnm0 := fmt.Sprintf("gb%d", i)
						gbnm := fmt.Sprintf("gbn_%d", ii)
						patgen.AddVocabRepeat(ss.PoolVocab, gbnm, 1, gbnm0, ii)
						fillseq = append(fillseq, gbnm)
						//fillseq = append(fillseq, "z") //empty
					}
				}
				fillt = []string{}
				for ii := 0; ii < nstates; ii++ { // all trials
					//testing
					if ii == i {
						fillt = append(fillt, "a")
					} else {
						//use empty
						fillt = append(fillt, "z")

						//use garbage
						//gbtnm0 := fmt.Sprintf("gbt%d", i)
						//gbtnm := fmt.Sprintf("gbn_%d", ii)
						//patgen.AddVocabRepeat(ss.PoolVocab, gbtnm, 1, gbtnm0, ii)
						//fillt = append(fillt, gbtnm) // empty pool

						//gb2nm := fmt.Sprintf("gb2_%d", ii)
						//patgen.AddVocabRepeat(ss.PoolVocab, gb2nm, 1, "garbaget", ii)
						//fillt = append(fillt, gb2nm) // empty pool
					}
				}
				//fmt.Printf("fillt: %s\n", fillt) //print sample cue sequence
				if i == 0 {
					//fmt.Printf("seqid: %s\n", fillseq) //print sample cue sequence
					//fmt.Printf("testseq: %s\n", fillt) //print sample test sequence
					patgen.VocabConcat(ss.PoolVocab, "A1", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT1", fillt)
				} else if i == 1 {
					patgen.VocabConcat(ss.PoolVocab, "A2", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT2", fillt)
				} else if i == 2 {
					patgen.VocabConcat(ss.PoolVocab, "A3", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT3", fillt)
				} else if i == 3 {
					patgen.VocabConcat(ss.PoolVocab, "A4", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT4", fillt)
				} else if i == 4 {
					patgen.VocabConcat(ss.PoolVocab, "B1", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT1", fillt)

				} else if i == 5 {
					patgen.VocabConcat(ss.PoolVocab, "B2", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT2", fillt)
				} else if i == 6 {
					patgen.VocabConcat(ss.PoolVocab, "B3", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT3", fillt)
				} else if i == 7 {
					patgen.VocabConcat(ss.PoolVocab, "B4", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT4", fillt)
				}
			}
		} else {
			for i := 0; i < ss.wpvc*2; i++ {
				fillseq = []string{}
				if i < ss.wpvc { //cue pool
					if seqid[0] == 1 { //initial
						fillseq = append(fillseq, "a")
					} else if seqid[0] == 2 {
						fillseq = append(fillseq, "b")
					} else if seqid[0] == 3 {
						fillseq = append(fillseq, "c")
					} else if seqid[0] == 4 {
						fillseq = append(fillseq, "d")
					} else if seqid[0] == 5 {
						fillseq = append(fillseq, "e")
					} else if seqid[0] == 6 {
						fillseq = append(fillseq, "f")
					} else if seqid[0] == 7 {
						fillseq = append(fillseq, "g")
					} else if seqid[0] == 8 {
						fillseq = append(fillseq, "h")
					} else if seqid[0] == 9 {
						fillseq = append(fillseq, "i")
					} else if seqid[0] == 10 {
						fillseq = append(fillseq, "j")
					} else if seqid[0] == 11 {
						fillseq = append(fillseq, "k")
					} else if seqid[0] == 12 {
						fillseq = append(fillseq, "l")
					} else if seqid[0] == 13 {
						fillseq = append(fillseq, "m")
					} else if seqid[0] == 14 {
						fillseq = append(fillseq, "n")
					} else if seqid[0] == 15 {
						fillseq = append(fillseq, "o")
					}
					for ii := 1; ii < 150-1; ii++ {
						if seqid[ii*2] == 1 {
							fillseq = append(fillseq, "a", "a")
						} else if seqid[ii*2] == 2 {
							fillseq = append(fillseq, "b", "b")
						} else if seqid[ii*2] == 3 {
							fillseq = append(fillseq, "c", "c")
						} else if seqid[ii*2] == 4 {
							fillseq = append(fillseq, "d", "d")
						} else if seqid[ii*2] == 5 {
							fillseq = append(fillseq, "e", "e")
						} else if seqid[ii*2] == 6 {
							fillseq = append(fillseq, "f", "f")
						} else if seqid[ii*2] == 7 {
							fillseq = append(fillseq, "g", "g")
						} else if seqid[ii*2] == 8 {
							fillseq = append(fillseq, "h", "h")
						} else if seqid[ii*2] == 9 {
							fillseq = append(fillseq, "i", "i")
						} else if seqid[ii*2] == 10 {
							fillseq = append(fillseq, "j", "j")
						} else if seqid[ii*2] == 11 {
							fillseq = append(fillseq, "k", "k")
						} else if seqid[ii*2] == 12 {
							fillseq = append(fillseq, "l", "l")
						} else if seqid[ii*2] == 13 {
							fillseq = append(fillseq, "m", "m")
						} else if seqid[ii*2] == 14 {
							fillseq = append(fillseq, "n", "n")
						} else if seqid[ii*2] == 15 {
							fillseq = append(fillseq, "o", "o")
						}
					}
					//final
					if seqid[len(seqid)-1] == 1 {
						fillseq = append(fillseq, "a")
					} else if seqid[len(seqid)-1] == 2 {
						fillseq = append(fillseq, "b")
					} else if seqid[len(seqid)-1] == 3 {
						fillseq = append(fillseq, "c")
					} else if seqid[len(seqid)-1] == 4 {
						fillseq = append(fillseq, "d")
					} else if seqid[len(seqid)-1] == 5 {
						fillseq = append(fillseq, "e")
					} else if seqid[len(seqid)-1] == 6 {
						fillseq = append(fillseq, "f")
					} else if seqid[len(seqid)-1] == 7 {
						fillseq = append(fillseq, "g")
					} else if seqid[len(seqid)-1] == 8 {
						fillseq = append(fillseq, "h")
					} else if seqid[len(seqid)-1] == 9 {
						fillseq = append(fillseq, "i")
					} else if seqid[len(seqid)-1] == 10 {
						fillseq = append(fillseq, "j")
					} else if seqid[len(seqid)-1] == 11 {
						fillseq = append(fillseq, "k")
					} else if seqid[len(seqid)-1] == 12 {
						fillseq = append(fillseq, "l")
					} else if seqid[len(seqid)-1] == 13 {
						fillseq = append(fillseq, "m")
					} else if seqid[len(seqid)-1] == 14 {
						fillseq = append(fillseq, "n")
					} else if seqid[len(seqid)-1] == 15 {
						fillseq = append(fillseq, "o")
					}
				} else { //target pool
					for ii := 0; ii < 150-1; ii++ {
						if seqid[ii*2+1] == 1 {
							fillseq = append(fillseq, "a", "a")
						} else if seqid[ii*2+1] == 2 {
							fillseq = append(fillseq, "b", "b")
						} else if seqid[ii*2+1] == 3 {
							fillseq = append(fillseq, "c", "c")
						} else if seqid[ii*2+1] == 4 {
							fillseq = append(fillseq, "d", "d")
						} else if seqid[ii*2+1] == 5 {
							fillseq = append(fillseq, "e", "e")
						} else if seqid[ii*2+1] == 6 {
							fillseq = append(fillseq, "f", "f")
						} else if seqid[ii*2+1] == 7 {
							fillseq = append(fillseq, "g", "g")
						} else if seqid[ii*2+1] == 8 {
							fillseq = append(fillseq, "h", "h")
						} else if seqid[ii*2+1] == 9 {
							fillseq = append(fillseq, "i", "i")
						} else if seqid[ii*2+1] == 10 {
							fillseq = append(fillseq, "j", "j")
						} else if seqid[ii*2+1] == 11 {
							fillseq = append(fillseq, "k", "k")
						} else if seqid[ii*2+1] == 12 {
							fillseq = append(fillseq, "l", "l")
						} else if seqid[ii*2+1] == 13 {
							fillseq = append(fillseq, "m", "m")
						} else if seqid[ii*2+1] == 14 {
							fillseq = append(fillseq, "n", "n")
						} else if seqid[ii*2+1] == 15 {
							fillseq = append(fillseq, "o", "o")
						}
					}
				}
				//create novel testing sequences
				//fillt := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"} // consistent order for testing
				/*fillt := []string{"c", "b", "a", "e", "d", "f", "j", "i", "h", "g", "o", "n", "m", "l", "k"} // re-arrange for presentation; consistent order for testing
				if i >= ss.wpvc {
					fillt = []string{"b", "a", "e", "d", "f", "j", "i", "h", "g", "o", "n", "m", "l", "k", "c"} // whip around to make this like a sequence
				}*/
				fillt = []string{"c", "c", "c", "c", "b", "b", "b", "b", "a", "a", "a", "a", "e", "e", "e", "e", "d", "d", "d", "d", "f", "f", "f", "f", "j", "j", "j", "j", "i", "i", "i", "i", "h", "h", "h", "h", "g", "g", "g", "g", "o", "o", "o", "o", "n", "n", "n", "n", "m", "m", "m", "m", "l", "l", "l", "l", "k", "k", "k", "k"}
				if i >= ss.wpvc {
					fillt = []string{"l", "a", "b", "e", "c", "a", "e", "d", "b", "e", "c", "d", "a", "b", "d", "c", "a", "b", "e", "f", "d", "h", "i", "j", "f", "g", "h", "i", "f", "g", "h", "j", "f", "g", "i", "j", "h", "i", "j", "o", "g", "l", "m", "n", "k", "l", "m", "o", "k", "l", "n", "o", "k", "m", "n", "o", "l", "m", "n", "c"} // whip around to make this like a sequence
				}
				if i == 0 {
					patgen.VocabConcat(ss.PoolVocab, "A1", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT1", fillt)
					fmt.Printf("seqid: %s\n", fillseq) //print sample cue sequence
				} else if i == 1 {
					patgen.VocabConcat(ss.PoolVocab, "A2", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT2", fillt)
				} else if i == 2 {
					patgen.VocabConcat(ss.PoolVocab, "A3", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT3", fillt)
				} else if i == 3 {
					patgen.VocabConcat(ss.PoolVocab, "A4", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "AT4", fillt)
				} else if i == 4 {
					patgen.VocabConcat(ss.PoolVocab, "B1", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT1", fillt)
					fmt.Printf("seqid: %s\n", fillseq) //print sample target sequence
				} else if i == 5 {
					patgen.VocabConcat(ss.PoolVocab, "B2", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT2", fillt)
				} else if i == 6 {
					patgen.VocabConcat(ss.PoolVocab, "B3", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT3", fillt)
				} else if i == 7 {
					patgen.VocabConcat(ss.PoolVocab, "B4", fillseq)
					patgen.VocabConcat(ss.PoolVocab, "BT4", fillt)
				}
			}
		}
		drf := float64(2)                                                                            //drift factor 1.65 (sqrt(e))
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "setTCSctxt", cvcn*2, plY, plX, pctAct, minDiff) //scramble; random starting point for temp cxts
		for j := 0; j < cvcn*2; j++ {
			drv := 1 / float32(math.Pow(drf, float64(j+2)))
			drvL := drv / float32(math.Pow(drf, 2)) //just make within-list drift a fraction of other drift
			ctxtNm1 := fmt.Sprintf("ctxtTCS_%d", j+1)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm1, ntrans, drvL, "setTCSctxt", j) //add drift during first learned list
		}
	}

	if exptype != 1 { //PI, regular A-B, A-C
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "C1", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "C2", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "C3", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "C4", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "C5", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "C6", npats, plY, plX, pctAct, minDiff)
	} else { //make B=C for spaced learning and L2=L1; archaic code that isn't used; exptype is basically 0 for main experiments
		patgen.AddVocabClone(ss.PoolVocab, "C1", "B1")
		patgen.AddVocabClone(ss.PoolVocab, "C2", "B2")
		patgen.AddVocabClone(ss.PoolVocab, "C3", "B3")
		patgen.AddVocabClone(ss.PoolVocab, "C4", "B4")
		patgen.AddVocabClone(ss.PoolVocab, "C5", "B5")
		patgen.AddVocabClone(ss.PoolVocab, "C6", "B6")
	}
	//////////////////////// ESTABLISH TIMING /////////////////////////////////
	//JWA reduced all lags by factor of 2 and fixed exponentially increasing to exp increasing away from AB in sp condition
	preablag := int(math.Pow(2, float64(8)))
	abaclag := npats
	testlag := npats
	halftime := int(math.Pow(2, float64(8)))       //relevant for exptype>0
	endtimef := 10                                 //10
	endtime := int(math.Pow(2, float64(endtimef))) //relevant for exptype>0
	fir := npats * 3                               //fir and las are for timing the PI/RI experiments
	las := halftime - npats*3
	fmt.Printf("exptype: %d\n", exptype)
	fmt.Printf("interval: %d\n", interval)
	fmt.Printf("npats: %d\n", npats)
	powoff := 5       //offset for the first testlag; before 8/5/21, was 4, tried 5 also
	if exptype == 0 { //fcurve
		preablag = halftime
		testlag = int(math.Pow(2, float64(endtimef-powoff+interval-1))) //-1 added when we made interval=0 the final learning temporal context
	} else if exptype < 3 { //PI/sp
		preablag = npats + halftime - int(math.Pow(2, float64((nints-1)-interval))) //JWA, fixed, 8/5/21 was 4+(4-interval), as above
		abaclag = halftime + npats - preablag
		testlag = endtime
	} else { //RI or RIn - all outdated!!!
		if interval == 0 {
			abaclag = fir
		} else if interval == 1 {
			abaclag = fir + (las-fir)/4
		} else if interval == 2 {
			abaclag = fir + (las-fir)/2
		} else if interval == 3 {
			abaclag = fir + (las-fir)*3/4
		} else if interval == 4 {
			abaclag = las
		}
		testlag = endtime - (abaclag)
	}
	ss.testlag = testlag + 0
	fmt.Printf("pre ab lag: %d\n", preablag)
	fmt.Printf("ab ac lag: %d\n", abaclag)
	fmt.Printf("test lag: %d\n", testlag)
	fmt.Printf("max epcs: %d\n", ss.MaxEpcs)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA1", npats, plY, plX, pctAct, minDiff) //lures
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA2", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA3", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA4", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA5", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA6", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB1", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB2", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB3", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB4", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB5", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB6", npats, plY, plX, pctAct, minDiff)

	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setctxt", cvcn*2, plY, plX, pctAct, minDiff)  // opening context in case we want drift at start
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setrctxt", cvcn*2, plY, plX, pctAct, minDiff) //scramble; random starting point for temp cxts
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setrctxt_mid2", cvcn*2, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setrctxt_mid3", cvcn*2, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setrctxt_mid4", cvcn*2, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setrctxt_mid5", cvcn*2, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "setrctxt_mid6", cvcn*2, plY, plX, pctAct, minDiff)

	//experiments
	ss.ece_start, ss.rawson_start, ss.cepeda_start, ss.cepeda_stop = 11, 15, 18, 27
	//0=no drift, 1-9 = fscales, 10=scramble ISIs
	//11-14 expanding/contracting/equal (ece)
	//15-17 Rawson override conditions
	//18-27  Cepeda
	//extras / modifications:
	//ran conditions lesioning particular pathways (e.g. ECin -> CA3)
	//also ran a condition of 16 with maxepcs=6 to test extra learning w/ low spacing

	//defaults
	fscale := ss.drifttype // settles fscale 1-9
	expand := float32(1)   //2=expand,1=constant,0.5=contract
	eqmatch := 0           //condition with equal spacing matching expanding/contracting
	fractioned := 1        //for reducing spacing in final ece condition
	blankouttc := 0        //0=ll temp contexts @ test, 1 = blank out shortest 2, 2 = no middle short, 3 = no 2 middle long, 4 = no 2 long
	if interval == 8 {     // for final RI, scramble temp context
		blankouttc = 5
	}
	maxfscale := 7         //maximum fscale value based on current training regime
	//expfact := 2		// if we do exp/contract, how many powers of 2 away
	if ss.drifttype == 0 { //no drift, overrides everything
		ss.driftbetween = 0
	} else if ss.drifttype == 10 { //scrambled tc for each training epoch
		ss.driftbetween = 2
	} else if ss.drifttype >= ece_start && ss.drifttype < rawson_start { //ece
		if ss.drifttype == ece_start { //expanding
			fscale = 6 //keep 4 for 16; 
			expand = float32(2)
		} else if ss.drifttype == ece_start+1 { //contracting
			fscale = 8 //256
			expand = float32(0.5)
		} else if ss.drifttype == ece_start+2 { //equal match for exp/contr
			eqmatch = 1
		} else if ss.drifttype == ece_start+3 { //equal for exp/contr, reduced overall spacing
			eqmatch = 1
			fractioned = 8
		}
	} else if ss.drifttype == rawson_start { //Rawson massed; see more below
		fscale = 1 //simulate ~17 trials
	} else if ss.drifttype == rawson_start+1 { //Rawson spaced; see more below
		fscale = 3 //simulate ~ 47 trials
	} else if ss.drifttype == rawson_start+2 { //Rawson massed with extra training trial (must have MaxEpcs = 6)
		fscale = 5 //run w/ maxepcs 6, for Massed+
	} else if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop { //Cepeda
		fscale = ss.drifttype - cepeda_start
	}
	if ss.drifttype >= cepeda_stop {
		fscale = 1
	}

	fmt.Printf("fscale: %d\n", fscale)
	fmt.Printf("expand: %v\n", expand)
	fmt.Printf("eqmatch: %v\n", eqmatch)
	fmt.Printf("blankouttc: %d\n", blankouttc)
	acrange := 1024   //time range for autocorrelation analysis
	drf := float64(2) //drift factor 1.65 (sqrt(e))
	//this loops over pools and changes drift values on each loop (along with various other complexities in different experiments)
	for i := 0; i < cvcn*2; i++ { //all pools, was (ecY-(lvc+wpvc))*ecX
		drv := float32(0)
		if ss.spect_type == 1 {
			drv = 1 / float32(math.Pow(drf, float64(i+2))) //spectral drift val /4/19/22
		} else if ss.spect_type == 2 {
			drv = 1 / float32(math.Pow(drf, float64(2))) //all pools @ FASTEST drift
		} else if ss.spect_type == 3 {
			drv = 1 / float32(math.Pow(drf, float64(cvcn+2))) //all pools @ medium drift
		} else if ss.spect_type == 4 {
			drv = 1 / float32(math.Pow(drf, float64(cvcn*2+2))) //all pools @ SLOWEST drift
		} else if ss.spect_type == 5 {
			drv = 1 / float32(math.Pow(1.05, float64(i+1))) //all pools down an order of mag
		}
		//drvL := drv + 0 // for now, keep same within-list and across-list!
		drvL := drv / float32(math.Pow(drf, 2)) //just make within-list drift a fraction of other drift
		filler := int(math.Pow(2, float64(fscale))) // time drift between epochs
		if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop {
			filler = int(math.Pow(2, 1))
		}
		filler2 := 0
		_ = filler2
		if ss.drifttype >= ece_start && ss.drifttype < rawson_start {
			filler2 = int(math.Pow(2, float64(1))) //impose 2
		}
		if expand == 0.5 {
			filler = int(math.Pow(2, float64(fscale))) //set initial filler so it contracts rather than expands
			fmt.Printf("fillerpost05: %v\n", filler)
		} else if eqmatch == 1 { //special condition whereby we set equal spacing for same timing as expanding/contracting intervals
			if ss.MaxEpcs == 4 {
				//FIX IF WE USE 4!!
				filler = 37 //16+32+64=112/3=37.3; for last one, add 1 to match exactly
			} else if ss.MaxEpcs == 5 {
				//filler = 136 //16+256/2=136 //JWA 5_8_23
				filler = 160 //64+256/2=160
			} else if ss.MaxEpcs == 6 {
				//FIX IF WE USE 6!!
				filler = 99 //16+32+64+128+256=240/5=99.2; for last one, add 1 to match exactly
			}
			filler /= fractioned //change in the final ECE experiment
			expand = 1           //enforce this to 1 so all other intervals are equal
		}
		ctxtNm0 := fmt.Sprintf("preab%d", i+1)
		patgen.AddVocabDrift(ss.PoolVocab, ctxtNm0, preablag, drv, "setctxt", i) // import from "setctxt", drift before AB list
		ctxtNm1 := fmt.Sprintf("ctxt_%d", i+1) //learning context name
		patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm0)
		patgen.AddVocabDrift(ss.PoolVocab, ctxtNm1, npats, drvL, "clone", preablag-1) //add drift during first learned list

		//pre-allocate; allows drift between epochs!
		fill1 := fmt.Sprintf("fill1")
		fill2 := fmt.Sprintf("fill2")
		fill3 := fmt.Sprintf("fill3")
		fill4 := fmt.Sprintf("fill4")
		fill5 := fmt.Sprintf("fill5")
		ctxtNm_1_2 := fmt.Sprintf("midctxt_2_%d", i+1) //various context names for later training epochs
		ctxtNm_1_3 := fmt.Sprintf("midctxt_3_%d", i+1)
		ctxtNm_1_4 := fmt.Sprintf("midctxt_4_%d", i+1)
		ctxtNm_1_5 := fmt.Sprintf("midctxt_5_%d", i+1)
		ctxtNm_1_6 := fmt.Sprintf("midctxt_6_%d", i+1)
		//between epoch 1 and epoch 2
		patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm1)
		//if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop {
		//	filler = int(math.Pow(2, float64(ss.drifttype-(cepeda_start-1)))) 
		//}
		if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop {
			filler = int(math.Pow(2, 1)) // 12_10_21
		}
		if ss.drifttype >= ece_start && ss.drifttype < rawson_start {
			//fmt.Printf("filler12: %v\n", filler2)
			patgen.AddVocabDrift(ss.PoolVocab, fill1, filler2+1, drv, "clone", npats-1) //+1 is to move it off the last trial of the previous pattern
			patgen.AddVocabClone(ss.PoolVocab, "clone", fill1)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_2, npats, drvL, "clone", filler2-1) //2nd learning context
			ss.fillers[0] = filler2 + 0
		} else {
			//fmt.Printf("filler12: %v\n", filler)
			patgen.AddVocabDrift(ss.PoolVocab, fill1, filler+1, drv, "clone", npats-1) //
			patgen.AddVocabClone(ss.PoolVocab, "clone", fill1)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_2, npats, drvL, "clone", filler-1)
			ss.fillers[0] = filler + 0
		}
		//between 2 and 3
		patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm_1_2)
		if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop {
			filler = int(math.Pow(2, 1)) // 12_10_21
		}
		if ss.drifttype >= ece_start && ss.drifttype < rawson_start { //JWA 5_8_23 fix - was <= ece_start+2 (instead of +3)
			patgen.AddVocabDrift(ss.PoolVocab, fill2, filler2+1, drv, "clone", npats-1)
			patgen.AddVocabClone(ss.PoolVocab, "clone", fill2)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_3, npats, drvL, "clone", filler2-1)
			ss.fillers[1] = filler2 + 0
		} else {
			patgen.AddVocabDrift(ss.PoolVocab, fill2, filler+1, drv, "clone", npats-1)
			patgen.AddVocabClone(ss.PoolVocab, "clone", fill2)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_3, npats, drvL, "clone", filler-1)
			ss.fillers[1] = filler + 0
		}

		//between 3 and 4
		patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm_1_3)
		//make TWO long delays
		if ss.drifttype >= rawson_start && ss.drifttype < rawson_start+3 {
			if ss.MaxEpcs == 5 {
				filler = int(math.Pow(2, float64(maxfscale)))
			}
		}
		if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop {
			filler = int(math.Pow(2, float64(ss.drifttype-(cepeda_start-1)))) //variable ISI for Cepeda
		}
		if eqmatch == 1 && ss.MaxEpcs == 4 { //add one to filler to accomodate the non-even divide for eqmatch
			ss.fillers[2] = filler + 1
			patgen.AddVocabDrift(ss.PoolVocab, fill3, filler+1+1, drv, "clone", npats-1)
		} else if eqmatch == 1 && ss.MaxEpcs == 6 { //add one to filler ""
			ss.fillers[2] = filler + 1
			patgen.AddVocabDrift(ss.PoolVocab, fill3, filler+1+1, drv, "clone", npats-1)
		} else {
			ss.fillers[2] = filler
			patgen.AddVocabDrift(ss.PoolVocab, fill3, filler+1, drv, "clone", npats-1)
		}
		patgen.AddVocabClone(ss.PoolVocab, "clone", fill3)
		patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_4, npats, drvL, "clone", filler-1) //learning context 4
		if ss.MaxEpcs > 4 {
			//between 4 and 5
			// in ece, if we were in contracting, we go from 2^8 (256) to 2^4 (16); if we were in expanding, we go from 2^4 (16) to 2^8 (256)
			//filler = int(float32(filler) * float32(math.Pow(float64(expand), 4))) //JWA before 5_8_23, this was '4' instead of 2 at end; test w/ 4 again on 6_8_23
			filler = int(float32(filler) * float32(math.Pow(float64(expand), 2))) //64/256 can you talk
			if ss.drifttype >= rawson_start && ss.drifttype < cepeda_start {
				filler = int(math.Pow(2, float64(maxfscale)))
			} else if ss.drifttype >= cepeda_start && ss.drifttype < cepeda_stop { // 2 rounds after delay
				filler = int(math.Pow(2, 1))
			}
			patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm_1_4)
			//fmt.Printf("filler45: %v\n", filler)
			ss.fillers[3] = filler
			patgen.AddVocabDrift(ss.PoolVocab, fill4, filler+1, drv, "clone", npats-1)
			patgen.AddVocabClone(ss.PoolVocab, "clone", fill4)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_5, npats, drvL, "clone", filler-1)
			if ss.MaxEpcs > 5 { //not used for ECE
				filler = int(float32(filler) * expand)
				if ss.drifttype > rawson_start && ss.drifttype < rawson_start+3 { //both spacing and massed can be here
					filler = int(math.Pow(2, float64(maxfscale)))
				}
				patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm_1_5)
				//fmt.Printf("filler56: %v\n", filler)
				ss.fillers[4] = filler
				patgen.AddVocabDrift(ss.PoolVocab, fill5, filler+1, drv, "clone", npats-1)
				patgen.AddVocabClone(ss.PoolVocab, "clone", fill5)
				patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_6, npats, drvL, "clone", filler-1)
			}
		}
		fmt.Printf("fillers epc: %v\n", ss.fillers)

		////// MUST touch this up if we run other exptypes!!
		lastctxt := ctxtNm_1_6 // just set sth
		if ss.driftbetween > 0 {
			if ss.MaxEpcs == 6 {
				lastctxt = ctxtNm_1_6
			} else if ss.MaxEpcs == 5 {
				lastctxt = ctxtNm_1_5
			} else if ss.MaxEpcs == 4 {
				lastctxt = ctxtNm_1_4
			} else if ss.MaxEpcs == 1 {
				lastctxt = ctxtNm1
			}
		} else if ss.driftbetween == 0 {
			lastctxt = ctxtNm1
		}
		//fmt.Printf("lastctxt: %v\n", lastctxt)
		ctxtNm4 := fmt.Sprintf("lagbeforetest_%d", i+1)
		ctxtNm5 := fmt.Sprintf("ctxtT_%d", i+1) //test context
		if exptype <= 1 {                       // no AC, use after last epoch!
			patgen.AddVocabClone(ss.PoolVocab, "clone", lastctxt)
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm4, testlag, drv, "clone", npats-1) // drift after AB, import from end of AB list
		}
		patgen.AddVocabClone(ss.PoolVocab, "clone", ctxtNm4)
		patgen.AddVocabDrift(ss.PoolVocab, ctxtNm5, npats, drvL, "clone", testlag-1) // drift within test, import from end of AC-test interval

		//create random "lesion" temporal contexts with same temporal drift to keep everything consistent, but start from a different initial vector
		ctxtNmr := fmt.Sprintf("r_%d", i+1)
		patgen.AddVocabDrift(ss.PoolVocab, ctxtNmr, npats, drvL, "setrctxt", i) // import from "setrctxt", which gives a random starting pattern
		if ss.driftbetween == 2 {                                               //override everything and randomly scramble ALL temporal contexts
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_2, npats, drvL, "setrctxt_mid2", i) // random starting pattern
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_3, npats, drvL, "setrctxt_mid3", i) // random starting pattern
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_4, npats, drvL, "setrctxt_mid4", i) // random starting pattern
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_5, npats, drvL, "setrctxt_mid5", i) // random starting pattern
			patgen.AddVocabDrift(ss.PoolVocab, ctxtNm_1_6, npats, drvL, "setrctxt_mid6", i) // random starting pattern
		}
	}

	//autocorr stuff
	//note: subspace([]int__) creates a SLICE given a particular row so we can compare
	if ss.Net.Nm == "stcm7zz" { //call "stcm7" only when we want to run / print these autocorr values - add "zz" to not do this!
		for f := 0; f < 100; f++ { //all "runs"
			patgen.AddVocabPermutedBinary(ss.PoolVocab, "setautoc", cvcn*2, plY, plX, pctAct, minDiff) // totally diff
			for i := 0; i < cvcn*2; i++ {                                                              //all drift values (simulating each of the different temporal context layers)
				drv := 1 / float32(math.Pow(2, float64(i+2))) //drift val
				//create drifting vectors for enumeration
				patnm := fmt.Sprintf("autocorr%d", i+1)
				patgen.AddVocabDrift(ss.PoolVocab, patnm, acrange, drv, "setautoc", i)       //random starting pattern
				fnm := ss.LogFileName("autocorr-" + strconv.Itoa(f) + "-" + strconv.Itoa(i)) //save
				file, err := os.Create(fnm)
				_ = err
				defer file.Close()
				for j := 0; j < acrange; j++ {
					autocorr := metric.Correlation32(ss.PoolVocab[patnm].SubSpace([]int{0}).(*etensor.Float32).Values, ss.PoolVocab[patnm].SubSpace([]int{j}).(*etensor.Float32).Values) // correlational difference
					writer := csv.NewWriter(file)
					defer writer.Flush()
					a1 := []string{fmt.Sprintf("%f", autocorr)}
					writer.Write(a1)
					/*if j == 500 { //to view a sample output
						fmt.Println(ss.PoolVocab[patnm].SubSpace([]int{0}).(*etensor.Float32).Values)
						fmt.Println(ss.PoolVocab[patnm].SubSpace([]int{i}).(*etensor.Float32).Values)
						fmt.Println(autocorr)
					}*/
				}
				writer := csv.NewWriter(file)
				defer writer.Flush()
				a1 := []string{fmt.Sprintf("autocorr%d", i+1)}
				writer.Write(a1)
			}
		}
	}

	//smithetal 1-4 spatial contexts, 5-8 pictorial contexts
	// 1 = old list in ph2, test w/ old list 2 = new list in ph2, test w/ old list,3-4 old/new in ph2,test w/ new list
	// 5-8 like 1-4 except new vector FOR EACH ITEM instead of EACH LIST
	smithetal := ss.smithetal //run list context vectors like smith et al. (1978)/smith & handy (2016), where they change
	fmt.Printf("smithetal: %d\n", smithetal)
	if smithetal > 0 { //replace ctxt_ vectors with list representations
		if smithetal < 5 { //same context for entire learning block like Smith et al. (1978)
			if smithetal == 1 || smithetal == 3 {
				patgen.AddVocabClone(ss.PoolVocab, "ctxt_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "ctxt_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_8", "ctxt_10")
			} else if smithetal == 2 || smithetal == 4 {
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff) //create new context for test
				patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_8", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_2_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_2_8", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_3_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_3_8", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_4_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_4_8", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_5_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_5_8", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_6_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "midctxt_6_8", npats, "q", 0)
			}
			patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
			patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_AC7", npats, "q", 0)
			patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
			patgen.AddVocabRepeat(ss.PoolVocab, "ctxt_AC8", npats, "q", 0)
			if smithetal == 1 || smithetal == 2 { //old test context
				patgen.AddVocabClone(ss.PoolVocab, "ctxtT_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "ctxtT_8", "ctxt_10")
			} else if smithetal == 3 || smithetal == 4 { //new test context
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "ctxtT_7", npats, "q", 0)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "q", 1, plY, plX, pctAct, minDiff)
				patgen.AddVocabRepeat(ss.PoolVocab, "ctxtT_8", npats, "q", 0)
			}
		} else { // different "pictorial" context for each trial (like Smith & Handy, 2016)
			patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt_9", npats, plY, plX, pctAct, minDiff) //ctxt_9, etc. to change on each trial
			patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt_10", npats, plY, plX, pctAct, minDiff)
			patgen.AddVocabClone(ss.PoolVocab, "ctxt_7", "ctxt_9")
			patgen.AddVocabClone(ss.PoolVocab, "ctxt_8", "ctxt_10")
			if smithetal == 5 || smithetal == 7 {
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_2_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_3_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_4_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_5_8", "ctxt_10")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "midctxt_6_8", "ctxt_10")
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt_AC7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt_AC8", npats, plY, plX, pctAct, minDiff)
			} else if smithetal == 6 || smithetal == 8 {
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_2_7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_2_8", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_3_7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_3_8", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_4_7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_4_8", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_5_7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_5_8", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_6_7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "midctxt_6_8", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt_AC7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt_AC8", npats, plY, plX, pctAct, minDiff)
			}

			if smithetal == 5 || smithetal == 6 { //old "pictorial" contexts at test
				patgen.AddVocabClone(ss.PoolVocab, "ctxtT_7", "ctxt_9")
				patgen.AddVocabClone(ss.PoolVocab, "ctxtT_8", "ctxt_10")
			} else if smithetal == 7 || smithetal == 8 { //new "pictorial" contexts
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxtT_7", npats, plY, plX, pctAct, minDiff)
				patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxtT_8", npats, plY, plX, pctAct, minDiff)
				//could alternatively test this with blanks
				//patgen.AddVocabEmpty(ss.PoolVocab, "ctxtT_7", npats, plY, plX)
				//patgen.AddVocabEmpty(ss.PoolVocab, "ctxtT_8", npats, plY, plX)
			}
		}
	}

	//////////////////////// MIX PATTERNS /////////////////////////////////
	patgen.InitPats(ss.TrainAB, "TrainAB_", "TrainAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	if ss.do_sequences == 0 {
		patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
		patgen.MixPats(ss.TrainAB, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
	} else if ss.do_sequences > 0 {
		if ss.seq_tce == 0 {
			patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT"})
			patgen.MixPats(ss.TrainAB, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT"})
		} else {
			patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
			patgen.MixPats(ss.TrainAB, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
		}
	}
	patgen.InitPats(ss.TrainAB2, "TrainAB2_", "TrainAB2 Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB2, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_2_1", "midctxt_2_2", "midctxt_2_3", "midctxt_2_4", "midctxt_2_5", "midctxt_2_6", "midctxt_2_7", "midctxt_2_8"})
	patgen.MixPats(ss.TrainAB2, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_2_1", "midctxt_2_2", "midctxt_2_3", "midctxt_2_4", "midctxt_2_5", "midctxt_2_6", "midctxt_2_7", "midctxt_2_8"})

	patgen.InitPats(ss.TrainAB3, "TrainAB3_", "TrainAB3 Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB3, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_3_1", "midctxt_3_2", "midctxt_3_3", "midctxt_3_4", "midctxt_3_5", "midctxt_3_6", "midctxt_3_7", "midctxt_3_8"})
	patgen.MixPats(ss.TrainAB3, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_3_1", "midctxt_3_2", "midctxt_3_3", "midctxt_3_4", "midctxt_3_5", "midctxt_3_6", "midctxt_3_7", "midctxt_3_8"})

	patgen.InitPats(ss.TrainAB4, "TrainAB4_", "TrainAB4 Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB4, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_4_1", "midctxt_4_2", "midctxt_4_3", "midctxt_4_4", "midctxt_4_5", "midctxt_4_6", "midctxt_4_7", "midctxt_4_8"})
	patgen.MixPats(ss.TrainAB4, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_4_1", "midctxt_4_2", "midctxt_4_3", "midctxt_4_4", "midctxt_4_5", "midctxt_4_6", "midctxt_4_7", "midctxt_4_8"})

	if ss.MaxEpcs > 4 {
		patgen.InitPats(ss.TrainAB5, "TrainAB5_", "TrainAB5 Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
		patgen.MixPats(ss.TrainAB5, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_5_1", "midctxt_5_2", "midctxt_5_3", "midctxt_5_4", "midctxt_5_5", "midctxt_5_6", "midctxt_5_7", "midctxt_5_8"})
		patgen.MixPats(ss.TrainAB5, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_5_1", "midctxt_5_2", "midctxt_5_3", "midctxt_5_4", "midctxt_5_5", "midctxt_5_6", "midctxt_5_7", "midctxt_5_8"})
		if ss.MaxEpcs > 5 {
			patgen.InitPats(ss.TrainAB6, "TrainAB6_", "TrainAB6 Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
			patgen.MixPats(ss.TrainAB6, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_6_1", "midctxt_6_2", "midctxt_6_3", "midctxt_6_4", "midctxt_6_5", "midctxt_6_6", "midctxt_6_7", "midctxt_6_8"})
			patgen.MixPats(ss.TrainAB6, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "midctxt_6_1", "midctxt_6_2", "midctxt_6_3", "midctxt_6_4", "midctxt_6_5", "midctxt_6_6", "midctxt_6_7", "midctxt_6_8"})
		}
	}

	//fmt.Printf("t context type: %s\n", ss.PoolVocab["ctxt_1"])
	//fmt.Printf("t context type 2: %f\n", ss.PoolVocab["ctxt_1"].SubSpace([]int{0}).(*etensor.Float32).Values) //all 49 from first pair
	patgen.AddVocabEmpty(ss.PoolVocab, "ee", npats, plY, plX)
	/*buff := ss.PoolVocab["ctxt_1"].SubSpace([]int{0}).(*etensor.Float32).Values
	buff2 := ss.PoolVocab["ctxtT_1"].SubSpace([]int{0}).(*etensor.Float32).Values
	buff3 := buff[0]*0.5 + buff2[0]*0.5
	fmt.Printf("b type: %f\n", buff)
	fmt.Printf("b2 type: %f\n", buff2)
	fmt.Printf("b3 type: %f\n", buff3)*/
	for hh := 0; hh < npats; hh++ { //loop through all list items - from pp.ListSize
		mmm := ss.PoolVocab["ee"].SubSpace([]int{hh}).(*etensor.Float32).Values
		buff := ss.PoolVocab["ctxt_1"].SubSpace([]int{hh}).(*etensor.Float32).Values
		buff2 := ss.PoolVocab["ctxtT_1"].SubSpace([]int{hh}).(*etensor.Float32).Values
		for h := 0; h < (plY * plX); h++ {
			mmm[h] = buff[h]*0.5 + buff2[h]*0.5
			///mmm = append(mmm, buff3)
		}
		//fmt.Printf("b type: %f\n", mmm)
		//ss.PoolVocab["ee"].SubSpace([]int{hh}).(*etensor.Float32).Values = mmm //does not work b/c .Values is a pointer to the original space...
	}
	//fmt.Printf("ee context type: %s\n", ss.PoolVocab["ee"])

	patgen.InitPats(ss.TrainABnc, "TrainABnc_", "TrainAB Pats, no temp context", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainABnc, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"})
	///////dilemma - when we train on ABnc, what do we want to insist is correct? original patterns? whatever it comes up with?
	//patgen.MixPats(ss.TrainABnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"})
	patgen.MixPats(ss.TrainABnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})

	if ss.do_sequences == 0 {
		patgen.InitPats(ss.TestAB, "TestAB_", "TestAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	}
	if blankouttc == 0 {
		if interval > 0 {
			if ss.do_sequences > 0 {
				patgen.InitPats(ss.TestAB, "TestAB_", "TestAB Pats", "Input", "ECout", ntrans, ecY, ecX, plY, plX)
				//patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
				patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT"})
			} else {
				patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
			}
		} else {
			if ss.driftbetween == 0 { //no drift condition, use original learning context
				if ss.do_sequences > 0 {
					patgen.InitPats(ss.TestAB, "TestAB_", "TestAB Pats", "Input", "ECout", ntrans, ecY, ecX, plY, plX)
					//patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
					patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT"})
				} else {
					patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
				}
			} else { //use last learned context (fix on 6/23/22)
				//patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
				patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "midctxt_5_1", "midctxt_5_2", "midctxt_5_3", "midctxt_5_4", "midctxt_5_5", "midctxt_5_6", "midctxt_5_7", "midctxt_5_8"})
			}
		}
	} else if blankouttc == 1 {
		patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "r_1", "r_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 2 {
		patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "r_3", "r_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 3 {
		patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "r_5", "r_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 4 {
		patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "r_7", "r_8"})
	} else if blankouttc == 5 {
		patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8"})
	}
	if ss.targortemp == 1 { //normal, testing target
		if ss.do_sequences > 0 {
			//patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "ctxtTCS_1", "ctxtTCS_2", "ctxtTCS_3", "ctxtTCS_4", "ctxtTCS_5", "ctxtTCS_6", "ctxtTCS_7", "ctxtTCS_8"})
			//patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
			patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"AT1", "AT2", "AT3", "AT4", "BT1", "BT2", "BT3", "BT4", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT", "emptyT"})
		} else {
			patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		}
	} else if ss.targortemp == 2 {
		patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
	}

	patgen.InitPats(ss.TestABnc, "TestABnc_", "TestAB Pats, no temp context", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestABnc, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"})
	if ss.targortemp == 1 {
		patgen.MixPats(ss.TestABnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
	} else if ss.targortemp == 2 {
		patgen.MixPats(ss.TestABnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_1", "ctxt_2", "ctxt_3", "ctxt_4", "ctxt_5", "ctxt_6", "ctxt_7", "ctxt_8"})
	}

	//if RIn condition (NON-specific RI condition), simply re-generate A1-4 so what is nominally A-C is really like D-C
	if exptype == 4 {
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "A1", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "A2", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "A3", npats, plY, plX, pctAct, minDiff)
		patgen.AddVocabPermutedBinary(ss.PoolVocab, "A4", npats, plY, plX, pctAct, minDiff)
	}

	if exptype != 1 {
		patgen.InitPats(ss.TrainAC, "TrainAC_", "TrainAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
		patgen.MixPats(ss.TrainAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8", "ctxt_AC9", "ctxt_AC10", "ctxt_AC11", "ctxt_AC12"})
		patgen.MixPats(ss.TrainAC, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8", "ctxt_AC9", "ctxt_AC10", "ctxt_AC11", "ctxt_AC12"})
		patgen.InitPats(ss.TestAC, "TestAC_", "TestAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)

		if blankouttc == 0 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if blankouttc == 1 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "r_1", "r_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if blankouttc == 2 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "r_3", "r_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if blankouttc == 3 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "r_5", "r_6", "ctxtT_7", "ctxtT_8"})
		} else if blankouttc == 4 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "r_7", "r_8"})
		} else if blankouttc == 5 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8"})
		}

		if ss.targortemp == 1 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if ss.targortemp == 2 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8", "ctxt_AC9", "ctxt_AC10", "ctxt_AC11", "ctxt_AC12"})
		}
		patgen.InitPats(ss.TestACnc, "TestACnc_", "TestAC Pats, no temp context", "Input", "ECout", npats, ecY, ecX, plY, plX)
		patgen.MixPats(ss.TestACnc, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"})
		if ss.targortemp == 1 {
			patgen.MixPats(ss.TestACnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if ss.targortemp == 2 {
			patgen.MixPats(ss.TestACnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8", "ctxt_AC9", "ctxt_AC10", "ctxt_AC11", "ctxt_AC12"})
		}
	} else { //sp
		patgen.InitPats(ss.TrainAC, "TrainAC_", "TrainAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
		patgen.MixPats(ss.TrainAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8"})
		patgen.MixPats(ss.TrainAC, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8"})
		patgen.InitPats(ss.TestAC, "TestAC_", "TestAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
		patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		if ss.targortemp == 1 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if ss.targortemp == 2 {
			patgen.MixPats(ss.TestAC, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8"})
		}

		patgen.InitPats(ss.TestACnc, "TestACnc_", "TestAC Pats, no temp context", "Input", "ECout", npats, ecY, ecX, plY, plX)
		patgen.MixPats(ss.TestACnc, ss.PoolVocab, "Input", []string{"A1", "A2", "A3", "A4", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"})
		if ss.targortemp == 1 {
			patgen.MixPats(ss.TestACnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
		} else if ss.targortemp == 2 {
			patgen.MixPats(ss.TestACnc, ss.PoolVocab, "ECout", []string{"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ctxt_AC1", "ctxt_AC2", "ctxt_AC3", "ctxt_AC4", "ctxt_AC5", "ctxt_AC6", "ctxt_AC7", "ctxt_AC8"})
		}
	}

	patgen.InitPats(ss.TestLure, "TestLure_", "TestLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	if blankouttc == 0 {
		patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 1 {
		patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "r_1", "r_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 2 {
		patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "r_3", "r_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 3 {
		patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "r_5", "r_6", "ctxtT_7", "ctxtT_8"})
	} else if blankouttc == 4 {
		patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "r_7", "r_8"})
	} else if blankouttc == 5 {
		patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8"})
	}
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "ECout", []string{"lA1", "lA2", "lA3", "lA4", "lB1", "lB2", "lB3", "lB4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8", "r_5", "r_6", "r_7", "r_8"})
	patgen.InitPats(ss.TestLurenc, "TestLurenc_", "TestLure Pats, no temp context", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestLurenc, ss.PoolVocab, "Input", []string{"lA1", "lA2", "lA3", "lA4", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"})
	patgen.MixPats(ss.TestLurenc, ss.PoolVocab, "ECout", []string{"lA1", "lA2", "lA3", "lA4", "lB1", "lB2", "lB3", "lB4", "ctxtT_1", "ctxtT_2", "ctxtT_3", "ctxtT_4", "ctxtT_5", "ctxtT_6", "ctxtT_7", "ctxtT_8", "r_5", "r_6", "r_7", "r_8"})

	ss.TrainAll = ss.TrainAB.Clone()
	ss.TrainAll.AppendRows(ss.TrainAC)
	ss.TrainAll.AppendRows(ss.TestAB)
	//ss.TrainAll.AppendRows(ss.TestAC)
	//ss.TrainAll.AppendRows(ss.TestLure)
	ss.EnvRSA(ss.TrainAll, "TrainAll")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		pnm := ss.ParamsName()
		if pnm == "Base" {
			return ss.Tag
		} else {
			return ss.Tag + "_" + pnm
		}
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur) //JWA, was TestEnv
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffCmp)
	dt.SetCellFloat("TrgOnWasOffAll", row, ss.TrgOnWasOffAll)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	//JWA, adding so we can find this in simmat!!
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
		tsr := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tsr, "Act")
		dt.SetCellTensor(lnm+"Act", row, tsr)
		ly.UnitValsTensor(tsr, "ActM")         //JWA
		dt.SetCellTensor(lnm+"ActM", row, tsr) //JWA
		// if we so choose
		if lnm == "CA1" {
			ss.Blankcontext(tsr)
			dt.SetCellTensor("IActM", row, tsr)
		}
	}

	lessLayStatNms := []string{"CA3"}
	//end of Q1 is before DG input, end of Q2 is after 1/4 DG input, end of ActM (Q3) is guess, end of ActP (Q4) is +
	//4 quarters are "ActQ1","ActQ2","ActM","ActP"
	for _, lnm := range lessLayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		tsrq1 := ss.ValsTsr(lnm + "Q1")
		tsrq4 := ss.ValsTsr(lnm + "ActP")
		ly.UnitValsTensor(tsrq1, "ActQ1") // create tsr1 name, etc.
		ly.UnitValsTensor(tsrq4, "ActP")
		//ED := metric.Correlation32(tsrq1.Values, tsrq2.Values) // correlational difference
		//ActPAvgEff is integrated over slow time constant, slowly progressing
		actavgg := ly.Pools[0].Inhib.Act.Avg                                                //average for this single trial, effective, most stable
		ED14 := metric.Abs32(tsrq1.Values, tsrq4.Values) / (actavgg * float32(tsrq4.Len())) // absolute difference, the error signal
		//fmt.Printf("DG err before each trial: %v\n", ss.DGED12)
		switch lnm {
		case "CA3":
			ss.CA3ED14 += float64(ED14)
		}
		//not currently using these, just aggregating above...
		dt.SetCellFloat(lnm+" ED14", row, float64(ED14))
	}

	lessLayStatNms = []string{"ECout"} //
	for _, lnm := range lessLayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		tsrq3 := ss.ValsTsr(lnm + "ActM")
		tsrq4 := ss.ValsTsr(lnm + "ActP")
		ly.UnitValsTensor(tsrq3, "ActM")
		ly.UnitValsTensor(tsrq4, "ActP")
		actavgg := ly.Pools[0].Inhib.Act.Avg                                                //average for this single trial, effective, most stable
		ED34 := metric.Abs32(tsrq3.Values, tsrq4.Values) / (actavgg * float32(tsrq4.Len())) // absolute difference, the error signal
		//fmt.Printf("DG err before each trial: %v\n", ss.DGED12)
		switch lnm {
		case "ECout":
			ss.ECoutED34 += float64(ED34)
		}
		//not currently using these, just aggregating above...
		dt.SetCellFloat(lnm+" ED34", row, float64(ED34))
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnTrlPlot != nil {
		ss.TrnTrlPlot.GoUpdate()
	}
}

func (ss *Sim) Blankcontext(tsr *etensor.Float32) {
	//loop over the y dimension of the pools and then the x dimension and just set everything to zero
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.CA1Pool.Y
	plX := hp.CA1Pool.X
	for i := 0; i < ecY; i++ {
		for ii := 0; ii < ecX; ii++ {
			for iii := 0; iii < plY; iii++ {
				for iiii := 0; iiii < plX; iiii++ {
					if i >= ss.wpvc {
						tsr.SetFloat([]int{i, ii, iii, iiii}, 0)
					}
				}
			}
		}
	}
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOffAll", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	//JWA, added for simmat
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm + "Act", etensor.FLOAT64, ly.Shp.Shp, nil})
		sch = append(sch, etable.Column{lnm + "ActM", etensor.FLOAT64, ly.Shp.Shp, nil}) //JWA
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Train Trial Plot"
	//plt.Params.XAxisCol = "Trial"
	plt.Params.XAxisCol = "TrialName" //JWA
	plt.Params.Type = eplot.Bar       //JWA
	plt.SetTable(dt)
	plt.Params.XAxisRot = 45 //JWA

	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("Mem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOnWasOff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOnWasOffAll", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOffWasOn", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	//JWA, added for simmat
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"Act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) //JWA
	}

	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.

//JWA, log TrnTrlLogLst

func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	ss.RepsAnalysisTrn()                   //JWA add to check training activations
	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view

	//7_23_21, grab weights
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()

	//ecin.SendPrjnVals(&ss.TmpValsWt, "Wt", ca3, 0, "qq") //another way, w/ send instead of rec
	//fmt.Printf("Random send values: %v\n", ss.TmpValsWt)
	// each output is ALL CA3 weights for a given ECin neuron. grab the average, log it, and move on!
	wtm := []float32{}
	for h := 0; h < (ecY * ecX * plY * plX); h++ {
		ca3.RecvPrjnVals(&ss.TmpValsWtR, "Wt", ecin, h, "qq") //grab rec values from ECin
		sum0 := float32(0)                                    //calculate means
		for i := 0; i < len(ss.TmpValsWtR); i++ {
			if !math.IsNaN(float64(ss.TmpValsWtR[i])) {
				sum0 += (ss.TmpValsWtR[i])
			}
		}
		sum0 /= float32(len(ss.TmpValsWtR))
		wtm = append(wtm, sum0)
	}
	//fmt.Printf("avg wts!: %v\n", wtm)
	//average within a pool!
	wtmp := []float32{}
	wtsp := []float32{}
	for h := 0; h < (ecY * ecX); h++ {
		poolavg := float32(0)
		for i := h * plY * plX; i < (h+1)*plY*plX; i++ {
			poolavg += wtm[i]
		}
		poolavg /= float32(plY * plX) //find mean in pool
		wtmp = append(wtmp, poolavg)

		//now calculate stdev
		stdev := float64(0)
		for i := h * plY * plX; i < (h+1)*plY*plX; i++ {
			stdev += math.Pow(float64(wtm[i]-poolavg), 2)
		}
		// The use of Sqrt math function func Sqrt(x float64) float64
		stdev = math.Sqrt(float64(stdev))
		wtsp = append(wtsp, float32(stdev))
	}
	fnm := ss.LogFileName("wts-" + strconv.Itoa(ss.runnum) + "-" + strconv.Itoa(epc) + "_epc_wt") //save
	file, err := os.Create(fnm)
	_ = err
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
	a1 := []string{fmt.Sprintf("%f", wtmp)}
	writer.Write(a1)
	fnm = ss.LogFileName("wts_s-" + strconv.Itoa(ss.runnum) + "-" + strconv.Itoa(epc) + "_epc_wt")
	file, err = os.Create(fnm)
	_ = err
	defer file.Close()
	writer = csv.NewWriter(file)
	defer writer.Flush()
	a1 = []string{fmt.Sprintf("%f", wtsp)}
	writer.Write(a1)

	//implement decay; not used but keep in case a reviewer asks / important later
	if ss.synap_decay == 1 {
		//fmt.Printf("epc val: %v\n", epc)
		//fmt.Printf("fillers epc: %v\n", ss.fillers[epc])
		r := rand.New(rand.NewSource(int64(epc)))
		ca3FmECin := ca3.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra() //grab projection
		/*fmt.Printf("Random send values prjn 1 val: %v\n", ca3FmECin.SynVal("Wt", 0, 0)) //note this may be nan so may want to print > 1
		fmt.Printf("Random send values prjn 1 val: %v\n", ca3FmECin.SynVal("Wt", 0, 1))
		fmt.Printf("Random send values prjn 1 val: %v\n", ca3FmECin.SynVal("Wt", 0, 2))
		fmt.Printf("Random send values prjn 1 val: %v\n", ca3FmECin.SynVal("Wt", 0, 3))
		cumu := float64(0)*/
		ndecays := 0
		if epc < ss.MaxEpcs-1 { //between epochs, not at final
			if ss.drifttype > 0 { //if not no drift condition
				ndecays = ss.Pat.ListSize + ss.fillers[epc]
			}
		} else { // at final epoch
			if ss.interval > 0 { //not the no RI drift either...
				ndecays = ss.Pat.ListSize + ss.testlag // drift the amount of actest
			}
		}
		if ndecays > 0 { //if in some kind of drift condition
			fmt.Printf("decaying!\n")
			for h := 0; h < (ecY * ecX * plY * plX); h++ { //ecin lay
				for i := 0; i < hp.CA3Size.Y*hp.CA3Size.X; i++ {
					sample := r.NormFloat64() * ss.decay_rate * float64(ndecays)
					ca3FmECin.SetSynVal("Wt", h, i, ca3FmECin.SynVal("Wt", h, i)+float32(sample))
					if ca3FmECin.SynVal("Wt", h, i) > 1 { //reset to 1
						ca3FmECin.SetSynVal("Wt", h, i, 1)
					} else if ca3FmECin.SynVal("Wt", h, i) < 0 { //reset to 0
						ca3FmECin.SetSynVal("Wt", h, i, 0)
					}
				}
			}
		}
		/*fmt.Printf("ndecays: %v\n", ndecays)
		fmt.Printf("cumu: %v\n", cumu)
		fmt.Printf("Random send values prjn 1 val post: %v\n", ca3FmECin.SynVal("Wt", 0, 0)) //note this may be nan so may want to print > 1
		fmt.Printf("Random send values prjn 1 val post: %v\n", ca3FmECin.SynVal("Wt", 0, 1))
		fmt.Printf("Random send values prjn 1 val post: %v\n", ca3FmECin.SynVal("Wt", 0, 2))
		fmt.Printf("Random send values prjn 1 val post: %v\n", ca3FmECin.SynVal("Wt", 0, 3))*/
	}

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.CntErr) / nt
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0

	trlog := ss.TrnTrlLog
	tix := etable.NewIdxView(trlog)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	mem := agg.Mean(tix, "Mem")[0]
	dt.SetCellFloat("Mem", row, mem)
	dt.SetCellFloat("TrgOnWasOff", row, agg.Mean(tix, "TrgOnWasOff")[0])
	dt.SetCellFloat("TrgOnWasOffAll", row, agg.Mean(tix, "TrgOnWasOffAll")[0])
	dt.SetCellFloat("TrgOffWasOn", row, agg.Mean(tix, "TrgOffWasOn")[0])

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	//fmt.Printf("Avg DG err: %v\n", ss.DGED12/float64(nt))
	dt.SetCellFloat("DG ED14", row, ss.DGED14/float64(nt))
	dt.SetCellFloat("CA3 ED14", row, ss.CA3ED14/float64(nt))
	dt.SetCellFloat("CA1 ED14", row, ss.CA1ED14/float64(nt))
	dt.SetCellFloat("ECout ED34", row, ss.ECoutED34/float64(nt))
	//reset to 0 after epoch
	ss.DGED14, ss.CA3ED14, ss.CA1ED14, ss.ECoutED34 = 0, 0, 0, 0

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnEpcPlot != nil {
		ss.TrnEpcPlot.GoUpdate()
	}
	if ss.TrnEpcFile != nil {
		if !ss.TrnEpcHdrs {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
			ss.TrnEpcHdrs = true
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func Float32ToByte(f float32) []byte {
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.LittleEndian, f)
	if err != nil {
		fmt.Println("binary.Write failed:", err)
	}
	return buf.Bytes()
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOffAll", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
		{"DG ED14", etensor.FLOAT64, nil, nil},    //JWA
		{"CA3 ED14", etensor.FLOAT64, nil, nil},   //JWA
		{"CA1 ED14", etensor.FLOAT64, nil, nil},   //JWA
		{"ECout ED34", etensor.FLOAT64, nil, nil}, //JWA
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("Mem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("TrgOnWasOff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOnWasOffAll", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOffWasOn", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DG ED14", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CA3 ED14", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CA1 ED14", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ECout ED34", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur

	row := dt.Rows
	if ss.TestNm == "AB" && trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestNm", row, ss.TestNm)
	dt.SetCellFloat("Trial", row, float64(row))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffCmp)
	dt.SetCellFloat("TrgOnWasOffAll", row, ss.TrgOnWasOffAll)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		tsr := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tsr, "Act")
		dt.SetCellTensor(lnm+"Act", row, tsr)
		ly.UnitValsTensor(tsr, "ActQ2")
		dt.SetCellTensor(lnm+"ActQ2", row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstTrlPlot != nil {
		ss.TstTrlPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestNm", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOffAll", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm + "Act", etensor.FLOAT64, ly.Shp.Shp, nil})
		sch = append(sch, etable.Column{lnm + "ActQ2", etensor.FLOAT64, ly.Shp.Shp, nil})
	}

	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Test Trial Plot"
	plt.Params.XAxisCol = "TrialName"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt) // this sets defaults so set params after
	plt.Params.XAxisRot = 45
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TestNm", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0) //JWA, was 0ff
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("Mem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)             //JWA, was On
	plt.SetColParams("TrgOnWasOff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)    //JWA, was On
	plt.SetColParams("TrgOnWasOffAll", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) //JWA, was On
	plt.SetColParams("TrgOffWasOn", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)    //JWA, was On

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
		plt.SetColParams(lnm+"Act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)   //JWA
		plt.SetColParams(lnm+"ActQ2", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) //JWA
	}

	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

// RepsAnalysis analyzes representations
//JWA this will run RSA on a partiular layer name in test trial log
//log is table, column in table, and then name, then true,
func (ss *Sim) RepsAnalysis() {
	acts := etable.NewIdxView(ss.TstTrlLog)
	epcnum, _, _ := ss.TrainEnv.Counter(env.Epoch) //grab epc //jwa 4/4/23
	epcnum -= 1                                    //jwa 4/4/23
	for _, lnm := range ss.LayStatNms {
		sm, ok := ss.SimMats[lnm]
		sm2, ok := ss.SimMatsQ2[lnm]
		if !ok {
			sm = &simat.SimMat{}
			ss.SimMats[lnm] = sm  //lnm
			sm2 = &simat.SimMat{} // always assign new so it doesn't just re-assign to old simmat
			ss.SimMatsQ2[lnm] = sm2
		}
		sm.TableCol(acts, lnm+"Act", "TrialName", true, metric.Correlation64)
		sm.Mat.SetMetaData("colormap", "Viridis")
		sm.Mat.SetMetaData("fix-max", "false")
		fnm := "simMat" + strconv.Itoa(ss.runnum) + "_" + strconv.Itoa(epcnum) + "_" + lnm + "_epc_rep.tsv" //jwa 4/4/23
		etensor.SaveCSV(ss.SimMats[lnm].Mat, gi.FileName(fnm), rune(etable.Tab))                            //jwa 4/4/23

		//sm2.TableCol(acts, lnm+"ActQ2", "TrialName", true, metric.Correlation64) // get initial response BEFORE full-loop recurrence
		//fnm2 := "simMat" + strconv.Itoa(ss.runnum) + "_" + strconv.Itoa(epcnum) + "_" + lnm + "_Q2_epc_rep.tsv"
		//etensor.SaveCSV(ss.SimMatsQ2[lnm].Mat, gi.FileName(fnm2), rune(etable.Tab))
	}
}

//note: tried the family trees thing
func (ss *Sim) RepsAnalysisTrn() {
	acts := etable.NewIdxView(ss.TrnTrlLog) //TrnTrlLog
	lnmtrn := ""
	epcnum, _, _ := ss.TrainEnv.Counter(env.Epoch) //grab epc
	epcnum -= 1                                    //subtract one b/c it's already added
	if epcnum == 0 {                               //JWA, fill in here so it doesn't cause issues by being blank
		ss.TrnTrlLogLst = ss.TrnTrlLog.Clone() // if epcnum==0, make this the base
	}
	lst := ss.TrnTrlLog.Clone()
	ss.TrnTrlLogLst.AppendRows(lst)             //append CURRENT epoch to end, so when epcnum==0, this is 2 copies, but otherwise it's 1
	actsL := etable.NewIdxView(ss.TrnTrlLogLst) //TrnTrlLogLst
	lsn := []string{"ECin", "DG", "CA3", "CA1"} //"ECin", Add ECout , "CA1", "DG", "ECin", "ECout"
	for _, lnm := range lsn {
		lnmtrn = lnm + "Trn"
		sm := &simat.SimMat{} // always assign new so it doesn't just re-assign to old simmat
		ss.SimMats[lnmtrn] = sm
		sm.TableCol(acts, lnm+"ActM", "TrialName", true, metric.Correlation64) //"Act"
		sm.Mat.SetMetaData("colormap", "Viridis")
		sm.Mat.SetMetaData("fix-max", "false")

		//this has same data as above PLUS the comparison w/ last epoch
		sm2 := &simat.SimMat{} // always assign new so it doesn't just re-assign to old simmat
		ss.SimMatsLst[lnmtrn] = sm2
		sm2.TableCol(actsL, lnm+"Act", "TrialName", true, metric.Correlation64) //"ActM"
		sm2.Mat.SetMetaData("colormap", "Viridis")
		sm2.Mat.SetMetaData("fix-max", "false")
		fnm := "simMatLst" + strconv.Itoa(ss.runnum) + "_" + strconv.Itoa(epcnum) + "_" + lnm + "_epc_rep.tsv"
		etensor.SaveCSV(ss.SimMatsLst[lnmtrn].Mat, gi.FileName(fnm), rune(etable.Tab))
	}

	//special item analysis
	lsn = []string{"CA1"} //"CA1" only before
	for _, lnm := range lsn {
		lnmtrn = lnm + "ItemTrn"
		sm3 := &simat.SimMat{} // always assign new so it doesn't just re-assign to old simmat
		ss.SimMatsItemLst[lnmtrn] = sm3
		sm3.TableCol(actsL, lnm+"ActM", "TrialName", true, metric.Correlation64) //
		//fmt.Printf("outputs: %v\n", ss.SimMatsItemLst[lnmtrn].Mat)
		sm3.Mat.SetMetaData("colormap", "Viridis")
		sm3.Mat.SetMetaData("fix-max", "false")
		//fnm := "simMatItemLst" + strconv.Itoa(ss.runnum) + "_" + strconv.Itoa(epcnum) + "_" + lnm + "_epc_rep.tsv"
		//etensor.SaveCSV(ss.SimMatsItemLst[lnmtrn].Mat, gi.FileName(fnm), rune(etable.Tab))
	}
	ss.TrnTrlLogLst = ss.TrnTrlLog.Clone() //JWA, log last either way
}

//JWA, new, check inputs
func (ss *Sim) EnvRSA(dt *etable.Table, nm string) {
	acts := etable.NewIdxView(dt)
	sm, ok := ss.SimMats[nm]
	if !ok {
		sm = &simat.SimMat{}
		ss.SimMats[nm] = sm
	}
	sm.TableCol(acts, "Input", "Name", true, metric.Correlation64)
}

// SimMatStat returns within, between for sim mat statistics
func (ss *Sim) SimMatStat(lnm string) (float64, float64) {
	sm := ss.SimMats[lnm]
	smat := sm.Mat
	nitm := smat.Dim(0)
	ncat := nitm / len(ss.TstNms)
	win_sum := float64(0)
	win_n := 0
	btn_sum := float64(0)
	btn_n := 0
	for y := 0; y < nitm; y++ {
		for x := 0; x < y; x++ {
			val := smat.FloatVal([]int{y, x})
			same := (y / ncat) == (x / ncat)
			if same {
				win_sum += val
				win_n++
			} else {
				btn_sum += val
				btn_n++
			}
		}
	}
	if win_n > 0 {
		win_sum /= float64(win_n)
	}
	if btn_n > 0 {
		btn_sum /= float64(btn_n)
	}
	return win_sum, btn_sum
}

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	ss.RepsAnalysis()

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		nt := ss.TrainAB.Rows * 4 // 1 train and 3 tests //JWA doubletrain...
		ss.EpcPerTrlMSec = float64(iv) / (float64(nt) * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	trix := etable.NewIdxView(trl)
	spl := split.GroupBy(trix, []string{"TestNm"})
	for _, ts := range ss.TstStatNms {
		split.Agg(spl, ts, agg.AggMean)
	}
	ss.TstStats = spl.AggsToTable(etable.ColNameOnly)

	for ri := 0; ri < ss.TstStats.Rows; ri++ {
		tst := ss.TstStats.CellString("TestNm", ri)
		for _, ts := range ss.TstStatNms {
			dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))
		}
	}

	for _, lnm := range ss.LayStatNms {
		win, btn := ss.SimMatStat(lnm)
		for _, ts := range ss.SimMatStats {
			if ts == "Within" {
				dt.SetCellFloat(lnm+" "+ts, row, win)
			} else {
				dt.SetCellFloat(lnm+" "+ts, row, btn)
			}
		}
	}

	// base zero on testing performance!
	curAB := ss.TrainEnv.Table.Table == ss.TrainAB //JWA doubletrain
	curAB2 := ss.TrainEnv.Table.Table == ss.TrainAB2
	curAB3 := ss.TrainEnv.Table.Table == ss.TrainAB3
	curAB4 := ss.TrainEnv.Table.Table == ss.TrainAB4
	curAB5 := ss.TrainEnv.Table.Table == ss.TrainAB5
	curAB6 := ss.TrainEnv.Table.Table == ss.TrainAB6
	var mem float64
	if curAB {
		mem = dt.CellFloat("AB Mem", row)
	} else if curAB2 {
		mem = dt.CellFloat("AB Mem", row)
	} else if curAB3 {
		mem = dt.CellFloat("AB Mem", row)
	} else if curAB4 {
		mem = dt.CellFloat("AB Mem", row)
	} else if curAB5 {
		mem = dt.CellFloat("AB Mem", row)
	} else if curAB6 {
		mem = dt.CellFloat("AB Mem", row)
	} else {
		mem = dt.CellFloat("AC Mem", row)
	}
	if ss.FirstZero < 0 && mem == 1 {
		ss.FirstZero = epc
	}
	if mem == 1 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstEpcPlot != nil {
		ss.TstEpcPlot.GoUpdate()
	}
	if ss.TstEpcFile != nil {
		if !ss.TstEpcHdrs {
			dt.WriteCSVHeaders(ss.TstEpcFile, etable.Tab)
			ss.TstEpcHdrs = true
		}
		dt.WriteCSVRow(ss.TstEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			sch = append(sch, etable.Column{lnm + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt) // this sets defaults so set params after
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	// JWA where we add columns for conditions
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "TrgOnWasOffAll" { //JWA was "Mem"
				plt.SetColParams(tn+" "+ts, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
			} else {
				plt.SetColParams(tn+" "+ts, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
			}
		}
	}
	// JWA where we add columns for HC layers
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			plt.SetColParams(lnm+" "+ts, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		}
	}
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		if ss.TstCycPlot != nil {
			ss.TstCycPlot.GoUpdate()
		}
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", eplot.On, eplot.FixMin, 0, eplot.FixMax, .5)
		plt.SetColParams(lnm+" Act.Avg", eplot.On, eplot.FixMin, 0, eplot.FixMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	epclog := ss.TstEpcLog
	epcix := etable.NewIdxView(epclog)
	if epcix.Len() == 0 {
		return
	}

	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	//JWA change this (I think) if we want to run with only a single epoch...
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		//nlast = epcix.Len() - 1
		nlast = epcix.Len() //JWA, I think the above was in error...
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	fzero := ss.FirstZero
	if fzero < 0 {
		fzero = ss.MaxEpcs
	}

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("NEpochs", row, float64(ss.TstEpcLog.Rows))
	dt.SetCellFloat("FirstZero", row, float64(fzero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			nm := tn + " " + ts
			dt.SetCellFloat(nm, row, agg.Mean(epcix, nm)[0])
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			nm := lnm + " " + ts
			dt.SetCellFloat(nm, row, agg.Mean(epcix, nm)[0])
		}
	}

	ss.LogRunStats()

	// note: essential to use Go version of update when called from another goroutine
	if ss.RunPlot != nil {
		ss.RunPlot.GoUpdate()
	}
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"NEpochs", etensor.FLOAT64, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			sch = append(sch, etable.Column{lnm + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("NEpochs", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "Mem" {
				plt.SetColParams(tn+" "+ts, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
			} else {
				plt.SetColParams(tn+" "+ts, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
			}
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			plt.SetColParams(lnm+" "+ts, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		}
	}
	return plt
}

//////////////////////////////////////////////
//  RunStats

// LogRunStats computes RunStats from RunLog data -- can be used for looking at prelim results
func (ss *Sim) LogRunStats() {
	dt := ss.RunLog
	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	for _, tn := range ss.TstNms {
		nm := tn + " " + "Mem"
		split.Desc(spl, nm)
	}
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "NEpochs")
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			split.Desc(spl, lnm+" "+ts)
		}
	}
	ss.RunStats = spl.AggsToTable(etable.AddAggName)
	if ss.RunStatsPlot != nil {
		ss.ConfigRunStatsPlot(ss.RunStatsPlot, ss.RunStats)
	}
}

func (ss *Sim) ConfigRunStatsPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Run Stats Plot"
	plt.Params.XAxisCol = "Params"
	plt.SetTable(dt)
	plt.Params.BarWidth = 10
	plt.Params.Type = eplot.Bar
	plt.Params.XAxisRot = 45

	cp := plt.SetColParams("AB Mem:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	cp.ErrCol = "AB Mem:Sem"
	cp = plt.SetColParams("AC Mem:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	cp.ErrCol = "AC Mem:Sem"
	cp = plt.SetColParams("FirstZero:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 30)
	cp.ErrCol = "FirstZero:Sem"
	cp = plt.SetColParams("NEpochs:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 30)
	cp.ErrCol = "NEpochs:Sem"
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("stcm7") //JWA, was "hip_bench"
	gi.SetAppAbout(`This demonstrates a basic Hippocampus model in Leabra. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("stcm7", "Hip PA learning + time", width, height) //JWA, was "hip_bench"
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	nv.SetNet(ss.Net)
	ss.NetView = nv
	nv.ViewDefaults()

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunStatsPlot").(*eplot.Plot2D)
	// ss.RunStatsPlot = ss.ConfigRunStatsPlot(plt, ss.RunStats)
	ss.RunStatsPlot = plt

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Pre Train", Icon: "fast-fwd", Tooltip: "Does full pretraining.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.PreTrain()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false, 1) // don't return on trial -- wrap, JWA: total guess with the second variable, will always test A-B
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Env", Icon: "gear", Tooltip: "select training input patterns: AB or AC."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SetEnv", vp)
		})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddAction(gi.ActOpts{Label: "Rebuild Net", Icon: "reset", Tooltip: "Rebuild network with current params"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.ReConfigNet()
		})

	tbar.AddAction(gi.ActOpts{Label: "Run Stats", Icon: "file-data", Tooltip: "compute stats from run log -- avail in plot"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.LogRunStats()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"SetEnv", ki.Props{
			"desc": "select which set of patterns to train on: AB or AC",
			"icon": "gear",
			"Args": ki.PropSlice{
				{"Train on AC", ki.Props{}},
			},
		}},
	},
}

// OuterLoopParams are the parameters to run for outer crossed factor testing
var OuterLoopParams = []string{"BigHip"} //, "SmallHip", "MedHip"}

// InnerLoopParams are the parameters to run for inner crossed factor testing
var InnerLoopParams = []string{"List016"} // ,"List020", "List040", "List050", "List060", "List070", "List080" "List100"}

// TwoFactorRun runs outer-loop crossed with inner-loop params
func (ss *Sim) TwoFactorRun() {
	tag := ss.Tag
	usetag := tag
	if usetag != "" {
		usetag += "_"
	}
	for _, otf := range OuterLoopParams {
		for _, inf := range InnerLoopParams {
			fmt.Printf("exp: %s\n", ss.pfix)
			fmt.Printf("RI: %d\n", ss.interval)
			ss.Tag = usetag + otf + "_" + inf
			rand.Seed(ss.RndSeed) // each run starts at same seed..
			ss.SetParamsSet(otf, "", ss.LogSetParams)
			ss.SetParamsSet(inf, "", ss.LogSetParams)
			ss.ReConfigNet() // note: this applies Base params to Network
			ss.ConfigEnv()
			ss.StopNow = false
			ss.PreTrain() //try no pretraining?
			ss.NewRun()
			ss.Train()
			fmt.Printf("exp done!\n")
		}
	}
	ss.Tag = tag
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	//var nogui bool
	var saveTrnEpcLog bool //JWA
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	// JWA, CHANGE THESE IN TERMINAL TO SET CUSTOM PARAMETERS
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	//flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	//flag.IntVar(&ss.MaxRuns, "runs", 20, "number of runs to do")
	//flag.IntVar(&ss.expnum, "expnum", 1, "which specific experiment # to run")
	//flag.IntVar(&ss.MaxEpcs, "epcs", 3, "maximum number of epochs to run") //JWA, was 30
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveTrnEpcLog, "trnepclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save test epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	//flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	//JWA add
	if saveTrnEpcLog {
		var err error
		fnm := ss.LogFileName("trn_epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving train epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	//end JWA add

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TstEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstEpcFile = nil
		} else {
			fmt.Printf("Saving test epoch log to: %v\n", fnm)
			defer ss.TstEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("running %d Runs\n", ss.MaxRuns)
	fmt.Printf("running expnum: %d\n", ss.expnum)
	fmt.Printf("pfix: %s\n", ss.pfix)
	fmt.Printf("RI: %d\n", ss.interval)
	// ss.Train()
	ss.TwoFactorRun()
	fnm := ss.LogFileName("runs")
	ss.RunStats.SaveCSV(gi.FileName(fnm), etable.Tab, etable.Headers)
}
