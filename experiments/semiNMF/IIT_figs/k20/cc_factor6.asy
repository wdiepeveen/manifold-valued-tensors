import settings;
import three;
surface ellipsoid(triple v1,triple v2,triple v3,real l1,real l2, real l3, triple pos=O) {
  transform3 T = identity(4);
  T[0][0] = l1*v1.x;
  T[1][0] = l1*v1.y;
  T[2][0] = l1*v1.z;
  T[0][1] = l2*v2.x;
  T[1][1] = l2*v2.y;
  T[2][1] = l2*v2.z;
  T[0][2] = l3*v3.x;
  T[1][2] = l3*v3.y;
  T[2][2] = l3*v3.z;
  T[0][3] = pos.x;
  T[1][3] = pos.y;
  T[2][3] = pos.z;
  return T*unitsphere;
}

size(200);

real gDx=1.5;
real gDy=1.5;
real gDz=1.5;

currentprojection=perspective(up=Y, camera = (gDx*-2.0,gDy*6.0,gDz*14.0), target = (gDx*1.5,gDy*1.5,gDz*0.0) );
currentlight=Viewport;

  draw(  ellipsoid( (0.5376179581793944,0.7327693769208334,0.41715221597166713), (-0.7480614833800612,0.6427834754401187,-0.16502551555545808), (-0.3890641953794277,-0.22333482475461763,0.8937284866924533), 0.14144807131257842, 0.26070191881881444, 0.9013634225028386,  (gDx*0, gDy*0, gDz*0)), rgb(0.12209621113716017,0.6323053572673212,0.5307626646555502)  );
  draw(  ellipsoid( (0.46619473630970065,0.77088583276935,0.43404780920151476), (-0.7742198337412225,0.5929060849436831,-0.2214633682541377), (-0.428072560296549,-0.23280336611024283,0.8732447972068873), 0.1275591065512793, 0.256307822526755, 0.9538736023450529,  (gDx*0, gDy*0, gDz*1)), rgb(0.12946242458319723,0.6502740850302636,0.5221829573282019)  );
  draw(  ellipsoid( (0.440716863974158,0.7693986366309608,0.4623790477078331), (-0.7713967252599809,0.5880348505156322,-0.24323261877724262), (-0.4590378394718245,-0.24948096626692295,0.8526684639433793), 0.13877035338251945, 0.2754949830725349, 0.9820938290700232,  (gDx*0, gDy*0, gDz*2)), rgb(0.12622633146735354,0.643871947302698,0.5254242714177618)  );
  draw(  ellipsoid( (0.4535206487132888,0.7367648481906055,0.5014943465896219), (-0.7495359558155904,0.6197359456111243,-0.2326439525478687), (-0.4821979594837454,-0.2703792081284342,0.8332947927843742), 0.1869103454751065, 0.3224595660901201, 1.0000000000000004,  (gDx*0, gDy*0, gDz*3)), rgb(0.1194580130830023,0.6096944482449502,0.5394682491077621)  );
  draw(  ellipsoid( (0.6447621851021198,0.6059404559683838,0.46595910602022467), (-0.6650189852150189,0.7452242378636839,-0.048893604940501594), (-0.3768706329311509,-0.27834690427830944,0.8834543151248712), 0.16244312209872147, 0.2864062558896254, 0.91296616646649,  (gDx*0, gDy*1, gDz*0)), rgb(0.11956634985499731,0.6162343334139124,0.5371729928010585)  );
  draw(  ellipsoid( (0.5158887043135172,0.709555898574191,0.479988824411874), (-0.7394191589927411,0.651753787272596,-0.1687492461917835), (-0.4325715571742888,-0.26785710290078024,0.8608916426295541), 0.12734188884520728, 0.2603985670537554, 0.9554555993370206,  (gDx*0, gDy*1, gDz*1)), rgb(0.12947471277511544,0.6502966444129014,0.5221712711335644)  );
  draw(  ellipsoid( (0.44538417334995783,0.74062042817375,0.5031046804603451), (-0.7547122114307868,0.6128731334009718,-0.23408545506401293), (-0.4817078119011829,-0.2754412890744725,0.831919275066017), 0.11167343270725794, 0.2498848373351687, 0.9687340999122254,  (gDx*0, gDy*1, gDz*2)), rgb(0.13953618113084695,0.6650124431692668,0.5139301931924292)  );
  draw(  ellipsoid( (0.41821216279259704,0.7339988633634441,0.5351114421066985), (-0.7443833406616401,0.614549239149096,-0.2611947066972905), (-0.5205689474390064,-0.28909323970379985,0.8033884923994158), 0.12189194802633548, 0.2591025802272974, 0.9620517394537663,  (gDx*0, gDy*1, gDz*3)), rgb(0.13254471461678807,0.6554274737384514,0.5194313177355704)  );
  draw(  ellipsoid( (-0.7608186791543329,-0.41637894363852207,-0.49777857802875025), (-0.5284680777567201,0.8427029670211158,0.10282606754089921), (-0.3766648752431967,-0.34129208116842824,0.8611871382514662), 0.2018540626039961, 0.3305634023571598, 0.8909238429104773,  (gDx*0, gDy*2, gDz*0)), rgb(0.12367344468409068,0.5808549306213796,0.5476351227725369)  );
  draw(  ellipsoid( (-0.5913941363835895,-0.6040257674274948,-0.5342338886056692), (0.6761448999654968,-0.7324507918576096,0.07964867706251649), (-0.43940988803366365,-0.3141157585839762,0.8415760455879531), 0.15777163646573358, 0.28771139885085295, 0.9376696780824749,  (gDx*0, gDy*2, gDz*1)), rgb(0.12023280309646389,0.6231603930964688,0.5345486410868144)  );
  draw(  ellipsoid( (0.4604683335701265,0.6976768092744795,0.5488314710179549), (-0.7385921190726326,0.6440854442267876,-0.19908697139436324), (-0.49239272478677976,-0.31368935323941544,0.8118795441697543), 0.1253531301684067, 0.26566905884171566, 0.9458582629844445,  (gDx*0, gDy*2, gDz*2)), rgb(0.1294534246869588,0.6502575624934824,0.5221915163204326)  );
  draw(  ellipsoid( (0.38710443328740474,0.7228564062281355,0.5723886561630848), (-0.7441227629457523,0.6114932694419052,-0.268993113468039), (-0.5444552060502469,-0.3217990015556212,0.774605661741888), 0.11348116615901582, 0.2545504177060768, 0.9230505936467877,  (gDx*0, gDy*2, gDz*3)), rgb(0.13420094841494123,0.6579022587289263,0.5180565890219218)  );
  draw(  ellipsoid( (-0.8565196818486845,-0.19027530988443372,-0.4797596700997394), (-0.35066695445104695,0.8965977550111478,0.2704536055684911), (-0.37869079954740825,-0.39988469857866843,0.8346768872927969), 0.24504641574431146, 0.38916581004326245, 0.8469400375126669,  (gDx*0, gDy*3, gDz*0)), rgb(0.13802275196793634,0.5368165263311505,0.5549779526299427)  );
  draw(  ellipsoid( (-0.7054970801872494,-0.41956497564798934,-0.5711734421843927), (-0.5480234830383917,0.833989244380897,0.06428220823408455), (-0.44938194432546485,-0.3583673694220534,0.8183084361338225), 0.208609494904381, 0.3269717985882931, 0.908108453236411,  (gDx*0, gDy*3, gDz*1)), rgb(0.1238618490050578,0.5801100058115471,0.5478053335431959)  );
  draw(  ellipsoid( (0.5109914397692878,0.6138717849766567,0.6017052269110545), (-0.6938454697365312,0.7077661118064853,-0.13283672347819372), (-0.5074112854564854,-0.34961201722466956,0.787594581497038), 0.16718951156623957, 0.2912503768425935, 0.9284137174370471,  (gDx*0, gDy*3, gDz*2)), rgb(0.11948553360746667,0.6148600830565975,0.5376762236016542)  );
  draw(  ellipsoid( (0.37590171728116983,0.6971574092829395,0.6104665802703433), (-0.7396985434249859,0.6225543979002333,-0.25548402398199455), (-0.5581612345908856,-0.3555243568828697,0.7497055874561696), 0.14086791814216063, 0.27755513374989704, 0.9128567758791039,  (gDx*0, gDy*3, gDz*3)), rgb(0.12224306264548793,0.6328822063788961,0.530514498211756)  );
  draw(  ellipsoid( (0.4837033245240445,0.75273361186854,0.4465682516791012), (-0.7938260708343202,0.5921954708787629,-0.13836435066311353), (-0.36860719351110177,-0.28757022417841444,0.883986483526669), 0.17493442194964096, 0.3023594919352667, 0.8378353090034876,  (gDx*1, gDy*0, gDz*0)), rgb(0.12117670992787692,0.5925841429366053,0.544682278402892)  );
  draw(  ellipsoid( (0.47841133998207636,0.7292116774439419,0.48925751834381304), (-0.7875536123009995,0.6027488578942448,-0.1282697238588687), (-0.3884351908983025,-0.3239508354466404,0.8626551794808996), 0.21342471029114984, 0.34173284423893313, 0.8804858633819976,  (gDx*1, gDy*0, gDz*1)), rgb(0.12666578369352804,0.5699299550430876,0.5499774487362086)  );
  draw(  ellipsoid( (0.5008357052038263,0.6837233701796845,0.5307409438352387), (-0.7642946582837572,0.637135930125907,-0.09955642551593545), (-0.40622317968316435,-0.35578105572490704,0.8416665424474054), 0.28191453605352734, 0.4034791724477429, 0.9035517824137875,  (gDx*1, gDy*0, gDz*2)), rgb(0.1423778717464556,0.5252954942990365,0.5560866200361666)  );
  draw(  ellipsoid( (0.5809645025967732,0.5820310691630924,0.5689640421426925), (-0.6952296079490636,0.718354258383247,-0.024959000253345263), (-0.4232446561428447,-0.381060354788481,0.821983556438328), 0.3956523045391195, 0.49742061598760995, 0.9310454732171238,  (gDx*1, gDy*0, gDz*3)), rgb(0.1693011748180046,0.4571072754132334,0.558041796650963)  );
  draw(  ellipsoid( (0.5053909523698763,0.7265818790749664,0.46546617305939686), (-0.78741280009718,0.6089636708433287,-0.09562598933415971), (-0.3529321004184628,-0.3181855128583156,0.8798846014684556), 0.13757833043622028, 0.2801899641385808, 0.8757827890593952,  (gDx*1, gDy*1, gDz*0)), rgb(0.12147949517157203,0.6298828281364005,0.5318048617004014)  );
  draw(  ellipsoid( (0.45730561719098056,0.732938301927037,0.5036596251973029), (-0.7954506751972765,0.5903607947651703,-0.1368661949950868), (-0.39765537317352123,-0.33804670915799284,0.8529915747612511), 0.14297785953818296, 0.289424154851969, 0.9152276383285506,  (gDx*1, gDy*1, gDz*1)), rgb(0.12169397407670736,0.6307253252090403,0.5314424107609291)  );
  draw(  ellipsoid( (0.43969747799787706,0.7155125057812594,0.542870133561361), (-0.7886143224188309,0.5968354104909812,-0.14790180276098763), (-0.4298297085104683,-0.36308311287456746,0.8266904347015998), 0.17131428224662884, 0.3216644063417811, 0.9364556507144829,  (gDx*1, gDy*1, gDz*2)), rgb(0.11943698394057777,0.6105632589943321,0.5391762039388103)  );
  draw(  ellipsoid( (0.4555516975461305,0.6726811504066389,0.5830717972517961), (-0.7636556589129718,0.6319063623705545,-0.13237969559493817), (-0.4574961043255271,-0.3849602824749772,0.8015627832202242), 0.2356981840571545, 0.3778177465557774, 0.9459324411640455,  (gDx*1, gDy*1, gDz*3)), rgb(0.12819818698083515,0.5649493369186935,0.5509213021861343)  );
  draw(  ellipsoid( (0.54168198469276,0.6747356284246483,0.5013107411513417), (-0.7697887414122686,0.6377447887641592,-0.026587177343096915), (-0.3376476285333984,-0.3715015694967555,0.8648587530962611), 0.1452886410446099, 0.2954951120151653, 0.8947012728069164,  (gDx*1, gDy*2, gDz*0)), rgb(0.12059175024154044,0.6255235155040015,0.5336090631020359)  );
  draw(  ellipsoid( (0.4584526536312419,0.706994302223082,0.5384981160622291), (-0.794188642375338,0.597854289952417,-0.10878716977432919), (-0.3988553180045243,-0.37779532105306535,0.8355747307631564), 0.12179530372440589, 0.28122018770598123, 0.9184480171455149,  (gDx*1, gDy*2, gDz*1)), rgb(0.12868653542455197,0.6488496622566902,0.5229208357553377)  );
  draw(  ellipsoid( (0.4166938235891199,0.7002630790099609,0.5796532390646746), (-0.7898972991882387,0.5945139062789582,-0.15038441400643962), (-0.4499205642308487,-0.3952022715223485,0.8008661876154911), 0.12155451033581414, 0.2851184961543257, 0.9263128376866459,  (gDx*1, gDy*2, gDz*2)), rgb(0.1292448438003343,0.6498746374819269,0.5223898788504902)  );
  draw(  ellipsoid( (-0.3996772140656458,-0.6742783754939461,-0.620972460659866), (0.7722375153829819,-0.6126601519844368,0.16821640230823348), (-0.4938697646058548,-0.41230596710653034,0.7655693600820933), 0.14536279769281785, 0.3039068144746557, 0.9188051484813976,  (gDx*1, gDy*2, gDz*3)), rgb(0.1211780596065222,0.6284948172483791,0.5323853176497559)  );
  draw(  ellipsoid( (-0.6128204387148461,-0.5779735242506607,-0.5388856234476982), (0.7176677144839677,-0.6925074011536779,-0.07339312593654175), (-0.3307629989595295,-0.43171762138306324,0.8391756275694906), 0.18162902239923673, 0.32575333513445015, 0.905514877173736,  (gDx*1, gDy*3, gDz*0)), rgb(0.12037819421814047,0.5978761625556168,0.5431840717754964)  );
  draw(  ellipsoid( (0.4832306969249479,0.6597822887144436,0.5754784314361052), (-0.7805867664404812,0.6223455479779141,-0.05805444832279133), (-0.3964497365460394,-0.4211571564283335,0.8157537961798131), 0.1451457597983647, 0.30088078474660473, 0.9207909572102424,  (gDx*1, gDy*3, gDz*1)), rgb(0.12130381554464445,0.6291158007487564,0.5321285518192234)  );
  draw(  ellipsoid( (0.4071887105593374,0.674578611280661,0.615744306669318), (-0.7903181648378407,0.5981584386225898,-0.13267886279258329), (-0.45781497607698685,-0.4326085753972524,0.77669509343911), 0.12289842020363022, 0.28965506497953303, 0.9115274166082767,  (gDx*1, gDy*3, gDz*2)), rgb(0.1275686130943827,0.6466769016977523,0.5240267547803654)  );
  draw(  ellipsoid( (0.3636708822410977,0.6580279439067718,0.6593502213905812), (-0.7757330218157971,0.6058088892635215,-0.17673106278702205), (-0.5157342031264875,-0.4472077981510676,0.7307690585939999), 0.11940493698775709, 0.2854868233874585, 0.8910152482283215,  (gDx*1, gDy*3, gDz*3)), rgb(0.1278399026739685,0.6472379673700132,0.5237463759985107)  );
  draw(  ellipsoid( (0.5849286434778947,0.6412798656611229,0.4966071041948174), (-0.7587690712688938,0.6489755767907408,0.05567941464190751), (-0.28657979434155884,-0.40937859571157076,0.8661877318736233), 0.2928873276905541, 0.4047790711124001, 0.7693608584035903,  (gDx*2, gDy*0, gDz*0)), rgb(0.15982070466258502,0.48066193056588935,0.5580908936749345)  );
  draw(  ellipsoid( (-0.63755278993396,-0.5621364164114231,-0.5268102973476761), (0.7160429112874472,-0.6846868453257328,-0.1359649698742044), (-0.2842692196535194,-0.463903624885622,0.8390378046164305), 0.3657027936146582, 0.4680653811472225, 0.8044998757491432,  (gDx*2, gDy*0, gDz*1)), rgb(0.17803975991662385,0.4360697307985035,0.5575128304581803)  );
  draw(  ellipsoid( (-0.7377168133471629,-0.42879914431999994,-0.5214452963977554), (-0.6116683269769065,0.751420114645633,0.24744629534348647), (-0.2857197246921861,-0.5014968645329504,0.8166181076766795), 0.46093047789300223, 0.5482873445809339, 0.826012027868788,  (gDx*2, gDy*0, gDz*2)), rgb(0.2069802888141928,0.37127674612230505,0.5530639010948724)  );
  draw(  ellipsoid( (-0.8782491275029706,-0.2135422979567608,-0.42787633379706413), (-0.3931254164153766,0.8318486744217376,0.3917654781001443), (-0.2722698606318865,-0.5122767512254918,0.8145192773319416), 0.5725766703243412, 0.6561248700250304, 0.8478502869606388,  (gDx*2, gDy*0, gDz*3)), rgb(0.24483536582879764,0.28799964396675676,0.5373523144265896)  );
  draw(  ellipsoid( (0.5330497629552268,0.6887806602568708,0.49136458182238313), (-0.7962638301361806,0.6047348201797985,0.01611552291942203), (-0.2860452115142563,-0.3998462195872512,0.8708048792074767), 0.20259445718419158, 0.345069973379271, 0.8216308663884233,  (gDx*2, gDy*1, gDz*0)), rgb(0.12819206426064939,0.5649687650850712,0.5509177530124179)  );
  draw(  ellipsoid( (0.5135008550911733,0.6675403011600479,0.5391723454961191), (-0.8015845959637323,0.5974083715633904,0.0237775755626585), (-0.30623358296289754,-0.4444020521027447,0.8418597322313117), 0.25272513977388666, 0.3914374920214601, 0.8702840465780969,  (gDx*2, gDy*1, gDz*1)), rgb(0.13798415298885314,0.5369196811917359,0.5549669643947934)  );
  draw(  ellipsoid( (0.535930023791255,0.6190038309621161,0.5741195579784136), (-0.7807916868831616,0.6220804074323801,0.058140419528164504), (-0.3211593861204308,-0.47942697456754346,0.8167046129193589), 0.33116423669306577, 0.4591725633356708, 0.8973144192473339,  (gDx*2, gDy*1, gDz*2)), rgb(0.1568160625099655,0.4882421886313964,0.5579648189261599)  );
  draw(  ellipsoid( (0.6135828909463015,0.5273998037042661,0.5876780436520331), (-0.7135765358693503,0.689029697673609,0.12667518770664096), (-0.3381191556162944,-0.49707899047990123,0.7991169587918945), 0.44142783582452516, 0.5447514740443655, 0.9085609865641977,  (gDx*2, gDy*1, gDz*3)), rgb(0.18590792191294042,0.41776656752532293,0.5567097188572743)  );
  draw(  ellipsoid( (-0.5080102020194001,-0.6989806324791001,-0.5033405507837864), (0.8151796185833068,-0.5789020605658527,-0.018830659016186437), (-0.27822261606595294,-0.4198791250964098,0.8638829181189375), 0.14675499181877094, 0.3110264862559654, 0.8324697872286682,  (gDx*2, gDy*2, gDz*0)), rgb(0.11946499970041367,0.6137141816453442,0.5380790064411061)  );
  draw(  ellipsoid( (0.47100072616163846,0.6819779490853437,0.559521575023303), (-0.8194542711795189,0.5730786541834142,-0.008692155369668674), (-0.3265777294930767,-0.45440833297891364,0.8287702054963556), 0.16641731332443568, 0.33073579026998295, 0.8798448568631274,  (gDx*2, gDy*2, gDz*1)), rgb(0.11968928369554016,0.6045780410801228,0.5411452094164977)  );
  draw(  ellipsoid( (0.4536497599525886,0.6558095779772413,0.6034199969575681), (-0.8161071857877772,0.5777227739780691,-0.014333796867523655), (-0.3580097157906667,-0.4859528720569486,0.7972947068299708), 0.21247811595031463, 0.37464448764857655, 0.9125585859087971,  (gDx*2, gDy*2, gDz*2)), rgb(0.12546177923481797,0.574082148743811,0.5491343223062206)  );
  draw(  ellipsoid( (-0.45773275630040156,-0.6154431354392984,-0.6416466869315877), (0.8007765676701394,-0.5989543998972808,0.003242763346083877), (-0.3863128427583892,-0.5123313126135608,0.7669934899564653), 0.2833159987624174, 0.43435114840695044, 0.9214577527705075,  (gDx*2, gDy*2, gDz*3)), rgb(0.1422923232640564,0.5255190869234889,0.5560681493411032)  );
  draw(  ellipsoid( (0.520309623863615,0.6678062966580903,0.5322712141932063), (-0.8117454954330587,0.5803391077960401,0.06538937686378603), (-0.26523036394621546,-0.46609148255069277,0.8440447760247412), 0.14386010403665636, 0.3172943113568654, 0.833905296003731,  (gDx*2, gDy*3, gDz*0)), rgb(0.11956857440394186,0.6162721610448079,0.5371591408643435)  );
  draw(  ellipsoid( (-0.4585298704392024,-0.6671341243712955,-0.5870966002408401), (0.823502340613253,-0.5673109276856354,0.0014853730028263389), (-0.33405725994131896,-0.4827943365738754,0.8095155190931732), 0.13231565436003348, 0.3114287819387786, 0.8618192296109843,  (gDx*2, gDy*3, gDz*1)), rgb(0.12175370975242819,0.6309599736090554,0.5313414625964867)  );
  draw(  ellipsoid( (-0.41757411023668534,-0.6462650214370357,-0.6387279424974521), (0.8217660361475624,-0.568554778844612,0.03802690215199556), (-0.3877272808229564,-0.5090058796516425,0.7684924008639864), 0.14645673387209715, 0.32578822181500033, 0.8866125539742891,  (gDx*2, gDy*3, gDz*2)), rgb(0.1200649136617454,0.62200641113159,0.5350049972772113)  );
  draw(  ellipsoid( (0.3936799320320126,0.6145352460315457,0.6836391902898908), (-0.813240274148876,0.579541567125282,-0.05264815738687604), (-0.42855147604171384,-0.5352363995085899,0.7279186967124486), 0.18337237104358797, 0.35679026061553615, 0.8957292471299574,  (gDx*2, gDy*3, gDz*3)), rgb(0.1211028743686507,0.5930240732422977,0.5445612754712011)  );
  draw(  ellipsoid( (-0.7825677574556593,-0.4602572989861018,-0.41922657802295704), (-0.6187150034290535,0.6497444431309994,0.4416151074772847), (-0.06913356292428866,-0.6049755179760627,0.7932371481007355), 0.401245812854787, 0.4915239858974857, 0.7155955511509937,  (gDx*3, gDy*0, gDz*0)), rgb(0.20899308674974182,0.3669596600171751,0.5525826751672187)  );
  draw(  ellipsoid( (-0.8619639411125224,-0.34916169456448765,-0.3675653346966002), (-0.5026390639910839,0.6831457320636335,0.5297790861419592), (-0.06612212619037136,-0.6414031647973824,0.7643492949012035), 0.46128220195373904, 0.551627102754868, 0.739151119898893,  (gDx*3, gDy*0, gDz*1)), rgb(0.22931368946813274,0.32329044733164386,0.5458873662877245)  );
  draw(  ellipsoid( (-0.9365878818395791,-0.2254268130154175,-0.2683018664954879), (-0.3469008988797802,0.7048442378102902,0.6187522660823406), (-0.04962767320408072,-0.6725900329319058,0.7383493357842253), 0.5268270856917676, 0.6313477352801921, 0.7569928989843484,  (gDx*3, gDy*0, gDz*2)), rgb(0.2521495319092574,0.2698962943385963,0.5316173094235679)  );
  draw(  ellipsoid( (-0.976738715973402,-0.16947774026131687,-0.13137266182333257), (-0.21424303965406852,0.7970621317749352,0.5646165761383392), (-0.009022232451845486,-0.5796285479911384,0.8148308693681588), 0.5957357096353602, 0.7254049066545686, 0.7797559913555905,  (gDx*3, gDy*0, gDz*3)), rgb(0.2680044481038186,0.22342031643345817,0.5119418766116544)  );
  draw(  ellipsoid( (-0.6622639045380144,-0.5807312352217245,-0.47345301053421734), (-0.7424670746347681,0.5935947490422755,0.3104640349424195), (-0.10074305846733644,-0.5571323957079576,0.8242901975781094), 0.334325332551012, 0.44428722616481303, 0.7913521835425864,  (gDx*3, gDy*1, gDz*0)), rgb(0.1701374121715977,0.4550633397226127,0.5580091230123178)  );
  draw(  ellipsoid( (-0.6919912846349,-0.5279528517762474,-0.492355408511645), (-0.7136185285052031,0.6033114790586331,0.35603912005586286), (-0.10907180095427968,-0.5977299101913939,0.7942432226334565), 0.40415728066600193, 0.5065823217871429, 0.8382218367832799,  (gDx*3, gDy*1, gDz*1)), rgb(0.18534107980594822,0.41906426459585,0.556777758598953)  );
  draw(  ellipsoid( (-0.7673774029468244,-0.43982211073390853,-0.4665709296089465), (0.630576471527944,-0.6495402562450727,-0.4248185130999511), (-0.11621202606423592,-0.6202047778166944,0.7757839896333194), 0.48292501409337163, 0.5771281839717306, 0.8585436618609812,  (gDx*3, gDy*1, gDz*2)), rgb(0.2086490718910142,0.3676961811907808,0.5526684958952736)  );
  draw(  ellipsoid( (-0.8815645473221349,-0.3018136319563337,-0.3629772451133078), (-0.4595285715403635,0.7246777949845076,0.513493510567042), (-0.108062208208016,-0.6194760891444636,0.7775422394542228), 0.5691131161066476, 0.6651328483318668, 0.8637411687254304,  (gDx*3, gDy*1, gDz*3)), rgb(0.24017381124286324,0.29894197243188125,0.5403471381487839)  );
  draw(  ellipsoid( (-0.5651970812593028,-0.6507147824465584,-0.5070725108320242), (-0.81534896538086,0.5341612718946502,0.2233333836672223), (-0.1255321631627774,-0.5396684236764439,0.8324660164224461), 0.245557824301796, 0.3801479019096355, 0.8042124799496833,  (gDx*3, gDy*2, gDz*0)), rgb(0.1418651642568407,0.526636649534727,0.5559744271145612)  );
  draw(  ellipsoid( (-0.5418799850121432,-0.6300856277365966,-0.5561997694740787), (0.8270537184491379,-0.5174791366775711,-0.2195392673370526), (-0.14949323945481105,-0.5789710224382691,0.8015262481878506), 0.30799069930636436, 0.43485181286098207, 0.8719409827438056,  (gDx*3, gDy*2, gDz*1)), rgb(0.15299210780265882,0.4979477335734456,0.5576887654003903)  );
  draw(  ellipsoid( (0.548952377062061,0.5961949954405923,0.5858351433035619), (-0.8185594737967371,0.525289599525722,0.23244617546373092), (-0.16914986129324003,-0.6071427871939994,0.7763800360537227), 0.38388670954982, 0.5019945506940691, 0.9119842282534609,  (gDx*3, gDy*2, gDz*2)), rgb(0.16932481398982144,0.45704932833810874,0.5580409879424535)  );
  draw(  ellipsoid( (0.611123396832276,0.5313529536897288,0.5866789858597132), (-0.7688524289165227,0.5746546171185682,0.2804247021456473), (-0.18813329421807157,-0.6224436598034501,0.7597168906751668), 0.4695889957006481, 0.5749714377591433, 0.9204131941371789,  (gDx*3, gDy*2, gDz*3)), rgb(0.19294787017489123,0.4018850511773503,0.5557441320611615)  );
  draw(  ellipsoid( (-0.5378265518658588,-0.6512588911476165,-0.5353545150731968), (-0.8305286269349185,0.5183588834656243,0.2037799493911762), (-0.1447922648130083,-0.5542255178476068,0.8196763235673407), 0.17705562588634724, 0.33872159591752937, 0.781142192153799,  (gDx*3, gDy*3, gDz*0)), rgb(0.12464422103533003,0.5770829506565007,0.5484863269341628)  );
  draw(  ellipsoid( (0.48393323090851037,0.6424536412076489,0.5941901605727613), (-0.8538269393100147,0.49545787742425484,0.15969048001288066), (-0.1918024653932853,-0.5846150961079309,0.7883127575218939), 0.21445166004444347, 0.3676163792197377, 0.8378363951591756,  (gDx*3, gDy*3, gDz*1)), rgb(0.13043197651412677,0.5580987631141817,0.5521036214898349)  );
  draw(  ellipsoid( (0.43905851497891735,0.6304603244415805,0.6401151456804666), (-0.8698415762439087,0.4766622371414,0.12715637585614104), (-0.22495166739106665,-0.6126278568497946,0.7576832163573419), 0.2698328340840291, 0.41759198164451594, 0.8918153972503763,  (gDx*3, gDy*3, gDz*2)), rgb(0.14113871129606648,0.5285470286931889,0.5558020308436699)  );
  draw(  ellipsoid( (0.4190014632099809,0.6076271422825384,0.6747051428508982), (-0.8722210172910959,0.4758515865649692,0.11311836526085078), (-0.252325723666021,-0.6358887666291619,0.7293676752168704), 0.3378642091560456, 0.4796905858780661, 0.9220147005286677,  (gDx*3, gDy*3, gDz*3)), rgb(0.15657180587529657,0.488860280129133,0.5579519280688128)  );