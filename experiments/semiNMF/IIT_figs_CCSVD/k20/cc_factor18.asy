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

  draw(  ellipsoid( (-0.6462344412459311,-0.3645014133790002,0.6704623528523211), (0.5273531802507999,-0.8483401927615438,0.04709077005819602), (-0.5516155094147782,-0.3840021314915047,-0.740447630007049), 0.03505933602961266, 1.068977420109597, 17.14047610042829,  (gDx*0, gDy*0, gDz*0)), rgb(0.512670128191561,0.8303474373079295,0.29635769007414203)  );
  draw(  ellipsoid( (-0.6726340638304655,-0.3738023506648739,0.63861977640246), (0.5391003412278366,-0.8387249941367023,0.07688436966262267), (-0.5068868101094599,-0.3959951853846247,-0.7656719760375542), 0.03321055454583902, 1.0329390323389085, 13.747101564792892,  (gDx*0, gDy*0, gDz*1)), rgb(0.502966719008804,0.8279612985512861,0.3024203316240725)  );
  draw(  ellipsoid( (-0.7141977623414577,-0.3760814130109715,0.590325611043707), (0.5320293627974594,-0.8397189574497121,0.1087052419200563), (-0.45482558567849135,-0.3917075992206954,-0.7998117549304814), 0.036313334751079736, 1.030047445515108, 10.480789659053253,  (gDx*0, gDy*0, gDz*2)), rgb(0.48017448430233567,0.8221409285330082,0.31654761239320317)  );
  draw(  ellipsoid( (-0.7702337722201813,-0.36690613079215256,0.5216510589643251), (0.5025539986944778,-0.8527645179595409,0.14223978102283066), (-0.39265686614098694,-0.3717157087027177,-0.8412181746586169), 0.042372017407857725, 1.053087378149022, 7.198898758483146,  (gDx*0, gDy*0, gDz*3)), rgb(0.4437041701291725,0.8121543208146165,0.3388095402631167)  );
  draw(  ellipsoid( (-0.6556070351588525,-0.3887871756065495,0.6473205910011678), (0.5340296542950431,-0.8448030368299008,0.03346875104473284), (-0.5338461798905522,-0.36763074007332375,-0.7614826952527579), 0.04908016702714072, 1.3499474623806014, 20.55927620254082,  (gDx*0, gDy*1, gDz*0)), rgb(0.5030691198135608,0.8279868563329207,0.3023565432188724)  );
  draw(  ellipsoid( (-0.6952170718871669,-0.3797552584506485,0.6102943278744405), (0.5266190970212752,-0.8469723948470335,0.0728703576193882), (-0.4892295469502336,-0.37205336451396576,-0.7888160396091775), 0.049537022370736934, 1.5980291298688925, 17.968883017564895,  (gDx*0, gDy*1, gDz*1)), rgb(0.49530485821445097,0.8260416672281533,0.3071891266840445)  );
  draw(  ellipsoid( (-0.7458064574756891,-0.37610431908241715,0.5498347653205594), (0.5098253340149043,-0.8535084292594056,0.10771021297785019), (0.42877833059070797,0.3606506652575231,0.8282996081522381), 0.05513485946971286, 1.8962309597389688, 15.815101289441092,  (gDx*0, gDy*1, gDz*2)), rgb(0.48165028943418375,0.8225260761650858,0.31563720708502463)  );
  draw(  ellipsoid( (-0.8002524156916042,-0.3751797503605345,0.4677993438421637), (0.47914193945833916,-0.86913263988628,0.12260283902262303), (0.36058157611260877,0.3222555029843757,0.8752898478586587), 0.06263175000088873, 2.1072965587744386, 12.916454732124995,  (gDx*0, gDy*1, gDz*3)), rgb(0.46077551842305636,0.8169361617577756,0.32844332380363683)  );
  draw(  ellipsoid( (-0.6733514592348262,-0.40244173039587827,0.6201922814597917), (0.529898091482735,-0.8476851604815845,0.02525631291905717), (0.5155635993641184,0.3456450814568056,0.7840431446517679), 0.0708966044849989, 1.7048756876509186, 19.9894185083898,  (gDx*0, gDy*2, gDz*0)), rgb(0.4775111294175256,0.8214458605967812,0.31819060195352394)  );
  draw(  ellipsoid( (-0.720419410813438,-0.39006399893149357,0.5734509126863349), (0.5155992915323471,-0.8542305802139335,0.06668947742116724), (-0.4638461416187053,-0.3437152783408655,-0.8165210127978956), 0.07427144448671585, 2.2250215607894246, 19.114963679486277,  (gDx*0, gDy*2, gDz*1)), rgb(0.4733998379232467,0.8203483698106392,0.32071499286122096)  );
  draw(  ellipsoid( (-0.7720675463119272,-0.3820787841791832,0.5078656383454558), (0.48998708122270457,-0.8667769255743946,0.09279235704870954), (-0.40475222564765995,-0.32048956920929333,-0.8564240023850963), 0.07970596489496361, 2.921561955664212, 17.89762486880339,  (gDx*0, gDy*2, gDz*2)), rgb(0.46740858066721047,0.8187452430266259,0.32439163240831126)  );
  draw(  ellipsoid( (-0.8243964416871807,-0.3825748087172088,0.41714148998691786), (0.4559346880833887,-0.8855639472945472,0.08888225612800045), (-0.3354013523166199,-0.2634634908048778,-0.9044848931171214), 0.08676484382870457, 3.6095129118362532, 16.411352142371335,  (gDx*0, gDy*2, gDz*3)), rgb(0.4591157971242392,0.8164834944070587,0.3294571358647226)  );
  draw(  ellipsoid( (-0.7093728006848719,-0.40446180839791923,0.5772355456795657), (0.5029453862988128,-0.8642274645934014,0.012523092397605002), (-0.49379769951796976,-0.2992014956345397,-0.8164816574552153), 0.10960913287742122, 2.0966039401902843, 17.71894274452241,  (gDx*0, gDy*3, gDz*0)), rgb(0.4374113455061497,0.8103444814041828,0.3426073354387832)  );
  draw(  ellipsoid( (-0.7571793440610313,-0.3975531321799781,0.5182961971895947), (0.4908808543145543,-0.8697928610177625,0.04996364468239359), (-0.43094712876955055,-0.29225311977192614,-0.853740409134332), 0.11510467543497237, 2.8550321505244867, 17.913803594516747,  (gDx*0, gDy*3, gDz*1)), rgb(0.43758192005839114,0.8103941406986686,0.3425046813850037)  );
  draw(  ellipsoid( (-0.8042195358773256,-0.3923606220543382,0.4464124554427127), (0.46091210703112284,-0.8859402735966431,0.051670699740143385), (0.3752212250198404,0.24731149160651267,0.8933342366740301), 0.11627104876632957, 3.958030521790901, 18.07578609885224,  (gDx*0, gDy*3, gDz*2)), rgb(0.44286036691616676,0.8119139132260338,0.33931988078290987)  );
  draw(  ellipsoid( (-0.8464213022814503,-0.39606213655145717,0.3559575298184723), (0.426976473410154,-0.9042158889219728,0.00920420421734804), (0.3182170174557007,0.15977612528579316,0.9344568045609474), 0.11537734583960813, 5.216239103114017, 18.838586285750033,  (gDx*0, gDy*3, gDz*3)), rgb(0.4520790964721758,0.814523995631345,0.33373595865644756)  );
  draw(  ellipsoid( (-0.6402415945485785,-0.39699222257747613,0.6576381039925295), (0.5564365163859846,-0.8298906374267152,0.040742277105119185), (0.5295933381776282,0.3920187560876086,0.7522314743703566), 0.03130047654824381, 0.6425604746401177, 11.423595322822761,  (gDx*1, gDy*0, gDz*0)), rgb(0.4937191255182597,0.8256370068978185,0.30817208581911193)  );
  draw(  ellipsoid( (-0.6832807303130972,-0.38345655328363804,0.6213602138265957), (0.5385894641328217,-0.8392842815809547,0.07431879853886839), (0.49299983033254724,0.3854386675399517,0.7799924364101912), 0.02558618703493484, 0.4807949341825001, 8.12133106649975,  (gDx*1, gDy*0, gDz*1)), rgb(0.48452097671607297,0.823275252587633,0.31386631682102367)  );
  draw(  ellipsoid( (0.7282962800132562,0.3720267492188002,-0.5754829505593927), (0.5204215760849733,-0.8466241987112606,0.11130520788241705), (-0.4458092772297194,-0.3805569129870162,-0.8102040016648322), 0.025872135519331162, 0.4260449911476727, 5.403136588566554,  (gDx*1, gDy*0, gDz*2)), rgb(0.45529302640100455,0.8154202075554058,0.33178225119809984)  );
  draw(  ellipsoid( (-0.7840549787119265,-0.3546384768842411,0.5094009629655856), (0.4849384941166201,-0.8622564649060667,0.14611107299447967), (-0.38741766516376747,-0.3615872501083507,-0.8480343231722027), 0.030003327230137662, 0.4253909590494388, 3.631361601293051,  (gDx*1, gDy*0, gDz*3)), rgb(0.41304232716424427,0.803080294178975,0.3571919500530089)  );
  draw(  ellipsoid( (-0.6527303895337025,-0.4128961843631734,0.6351848388599285), (0.5652619979536574,-0.8236575849651147,0.045464891937344584), (-0.5044025299784112,-0.38872216770820545,-0.7710208583972368), 0.03624214559793656, 0.8867926212323017, 16.91755646424425,  (gDx*1, gDy*1, gDz*0)), rgb(0.509280297872506,0.8295202922079605,0.2984789057752851)  );
  draw(  ellipsoid( (-0.7064385724251887,-0.3896811170118742,0.5908410703685377), (0.5357908798685167,-0.839889685870359,0.08668015123608436), (-0.46246370283470256,-0.37780145925203373,-0.8021180592328352), 0.03526512167700447, 0.8549997920297561, 14.020334978013096,  (gDx*1, gDy*1, gDz*1)), rgb(0.4994368556691746,0.8270802949629983,0.30461918497824303)  );
  draw(  ellipsoid( (-0.7557056526895,-0.37878457223808426,0.5342576291711223), (0.5124715300707204,-0.8499534240856006,0.12227881154205551), (0.40777677393411166,0.36619861375976603,0.8364309164063187), 0.03877516837021597, 0.8671472089240876, 10.38217998464933,  (gDx*1, gDy*1, gDz*2)), rgb(0.47371433244627176,0.8204323259720605,0.3205218903623198)  );
  draw(  ellipsoid( (-0.8090706637129385,-0.36704937858272346,0.4589982732006102), (0.4770917169256682,-0.8662618555895868,0.1482359308379862), (-0.3432027895574629,-0.3389176171763905,-0.8759832726755993), 0.04450744212629983, 0.888991841568632, 7.359066672322465,  (gDx*1, gDy*1, gDz*3)), rgb(0.4394211029569004,0.8109295812082302,0.341397836444225)  );
  draw(  ellipsoid( (-0.6619666239263354,-0.42879288399975424,0.6147656882412541), (0.5719974600416522,-0.8190396366558373,0.044642796648199874), (-0.48437495240099654,-0.3811964535745946,-0.7874453436707112), 0.04643185301873249, 1.1819005944314511, 20.318563756438436,  (gDx*1, gDy*2, gDz*0)), rgb(0.5054508324773944,0.8285812978739147,0.3008729059287979)  );
  draw(  ellipsoid( (-0.7179183753568913,-0.40053502729664103,0.5693548087382775), (0.5441287001365851,-0.8330093630085832,0.10009674733828658), (-0.43418563312550107,-0.38166358622288465,-0.8159753323102938), 0.049763474036654966, 1.3763119976275873, 18.823793027887223,  (gDx*1, gDy*2, gDz*1)), rgb(0.49687414854408,0.826440679783682,0.30621556892164026)  );
  draw(  ellipsoid( (-0.7693702925249086,-0.3865722453726952,0.5085580125095357), (0.517081661713708,-0.8443327293943483,0.14045923676597763), (0.3750945322175822,0.37103118626471676,0.8494939380121967), 0.05565066586002833, 1.5705768650072334, 15.73093837066436,  (gDx*1, gDy*2, gDz*2)), rgb(0.4788078008151967,0.8217842588768918,0.3173907019545825)  );
  draw(  ellipsoid( (-0.8206377523791917,-0.38007658147430323,0.42672645990708896), (0.48183306562295714,-0.8616948249152978,0.15911921815600255), (-0.30723049367276656,-0.3361901559073408,-0.8902727687785279), 0.06228628403951702, 1.6910554189805527, 12.258871911182538,  (gDx*1, gDy*2, gDz*3)), rgb(0.45480015359751114,0.8152827688142641,0.33208186235749837)  );
  draw(  ellipsoid( (-0.6731942670771327,-0.44647258159254377,0.5894673126311342), (0.5777919765024639,-0.8150763857404923,0.04250784983496777), (-0.4614822972377797,-0.369205524463415,-0.8066730254829625), 0.0644336316386255, 1.5238706062359755, 21.415918019658278,  (gDx*1, gDy*3, gDz*0)), rgb(0.4880706775422753,0.8241955894197464,0.31167342847937385)  );
  draw(  ellipsoid( (-0.7292707564551292,-0.4181587937335464,0.5415786064854015), (0.5533696896979508,-0.8259828734781581,0.10739776182192454), (-0.4024253350659697,-0.37801523242333784,-0.8337615568931995), 0.06954617087479499, 1.9048834645417372, 20.44404061356811,  (gDx*1, gDy*3, gDz*1)), rgb(0.48106845178019625,0.8223742313246993,0.3159961352912337)  );
  draw(  ellipsoid( (-0.7802772009731065,-0.4022677352477728,0.47890307873326743), (0.5264021995137712,-0.8358873648186511,0.15554111251882943), (-0.3377398614150217,-0.37346081791154395,-0.8639784739773338), 0.077683492535804, 2.4098782447289833, 18.576937472247348,  (gDx*1, gDy*3, gDz*2)), rgb(0.46910233868845413,0.819201127130565,0.3233536964967623)  );
  draw(  ellipsoid( (0.8278701775714374,0.39772969285577264,-0.39552757237478364), (0.4917326009274915,-0.8538678042787958,0.1706130768762828), (-0.2698703730921967,-0.33573928015196636,-0.9024683470848814), 0.08354425125257879, 2.8974553734830386, 16.678084345279366,  (gDx*1, gDy*3, gDz*3)), rgb(0.4591844854561595,0.8165022282542761,0.32941517878469584)  );
  draw(  ellipsoid( (-0.554072869737518,-0.5047320699691407,0.6620036197525644), (-0.6380789778295851,0.7682347861255796,0.05167718489369116), (-0.5346573417365411,-0.3937776668771977,-0.7477169758645086), 0.04380776710314622, 0.4909504058401332, 5.2768871547942355,  (gDx*2, gDy*0, gDz*0)), rgb(0.4119044334387402,0.8027279731700578,0.35786687445501963)  );
  draw(  ellipsoid( (-0.6211565341680471,-0.46778372760622045,0.6287630270993747), (0.5962325921766739,-0.8027696542130051,-0.00822060222953589), (0.5085973418005876,0.36978272872263973,0.7775535206349561), 0.030224558271447564, 0.2838739392158426, 3.494912630925085,  (gDx*2, gDy*0, gDz*1)), rgb(0.40868281689801883,0.8017280863487343,0.35977663659422876)  );
  draw(  ellipsoid( (-0.701471790871954,-0.41920328269523055,0.5763730861069383), (0.5348489616174743,-0.8441379258222801,0.036983110268991294), (0.4710349401770564,0.33421515520095924,0.8163493830256723), 0.025456519395928568, 0.2014938795172562, 2.3329070433235346,  (gDx*2, gDy*0, gDz*2)), rgb(0.38872914468313163,0.7953876661249223,0.3715386797757943)  );
  draw(  ellipsoid( (-0.7833583564102902,-0.36924663297532456,0.5000066094348701), (0.4621296159700863,-0.8839465804953425,0.07123665400572718), (-0.4156752379989805,-0.2868716905987365,-0.8630867451464619), 0.025536662271312672, 0.17508828363628742, 1.5386893712593566,  (gDx*2, gDy*0, gDz*3)), rgb(0.34915898191040645,0.7818590043515365,0.39446907154659927)  );
  draw(  ellipsoid( (-0.5693686533283993,-0.5275670878274751,0.6304699076468616), (-0.6516938090155258,0.7571442626646709,0.04503048750355081), (-0.5011132765132462,-0.3852343875554319,-0.7749064141862231), 0.04216787277519782, 0.6258618273284379, 8.921283415166798,  (gDx*2, gDy*1, gDz*0)), rgb(0.4560920373376825,0.8156430136354272,0.3312965425296293)  );
  draw(  ellipsoid( (-0.6586132614304514,-0.4754999513866632,0.5832052538336949), (0.592608927441788,-0.805392094185258,0.012579099338904108), (-0.46372753960082763,-0.3538974015942551,-0.812227430071527), 0.03531918851793713, 0.4600969769625589, 6.699585071237682,  (gDx*2, gDy*1, gDz*1)), rgb(0.4480021543061231,0.8133788575262815,0.33621007696987487)  );
  draw(  ellipsoid( (-0.7444443758889003,-0.41806326689563833,0.5206012640014738), (0.5201325983572411,-0.852004812777975,0.05958086210615438), (0.4186462126109952,0.3151363258586892,0.8517185243909839), 0.03343671499887784, 0.3898487053716959, 4.6627873164523015,  (gDx*2, gDy*1, gDz*2)), rgb(0.4239605665256484,0.8063844547716501,0.350680948044036)  );
  draw(  ellipsoid( (-0.8220273178596555,-0.36699678075791115,0.4354129667405306), (0.444330705616231,-0.8915926071067275,0.08736616621559373), (-0.3561478804359808,-0.26528472603216197,-0.895981417996489), 0.035156503439671784, 0.3772677232317885, 3.261803078168062,  (gDx*2, gDy*1, gDz*3)), rgb(0.38950695534566093,0.7956395803207374,0.3710822799487593)  );
  draw(  ellipsoid( (-0.5925551673939565,-0.5308015687414074,0.6059109408290805), (0.6536065313862487,-0.7564696767604852,-0.023497452439013067), (0.47082573817067086,0.3821038115009432,0.795185388140712), 0.04886530107426303, 0.831640602381708, 13.47900736702535,  (gDx*2, gDy*2, gDz*0)), rgb(0.47497121621581395,0.8207678584819065,0.3197501522907239)  );
  draw(  ellipsoid( (-0.6905153530866387,-0.4732849736606161,0.5469825233576525), (0.5851623067181253,-0.8100337908051394,0.03781973757416196), (-0.42517481339792124,-0.3461886645471314,-0.8362892959921921), 0.04621183100631877, 0.7486044864172536, 11.485039337187688,  (gDx*2, gDy*2, gDz*1)), rgb(0.467685761629707,0.8188208405212486,0.32422232120944755)  );
  draw(  ellipsoid( (-0.7765383531109454,-0.4145034601393957,0.4745261506810864), (0.5083304280193397,-0.857139663279479,0.08313707706442615), (-0.37227457890301324,-0.30577521018815323,-0.8763065438172734), 0.047242891268384296, 0.7453852061877262, 8.721902424760044,  (gDx*2, gDy*2, gDz*2)), rgb(0.4461471480228263,0.8128503485321237,0.3373320030970167)  );
  draw(  ellipsoid( (-0.8460208814241146,-0.3655985027683658,0.38805463915264976), (0.43586148823381193,-0.8934497504450342,0.1085002603882264), (-0.30703978776205937,-0.2609315584655018,-0.9152274528923364), 0.04940262714745301, 0.7685392602950533, 6.419404364521562,  (gDx*2, gDy*2, gDz*3)), rgb(0.4192828730691739,0.8049763933405283,0.3534739861281042)  );
  draw(  ellipsoid( (-0.6123544434022882,-0.5333135403056158,0.583608347586088), (-0.6553714223996808,0.7553024232933434,0.0025589195028458186), (-0.4421655056054159,-0.38091326715394525,-0.812031248511296), 0.060886606536926, 1.0854646697653283, 16.57116198658255,  (gDx*2, gDy*3, gDz*0)), rgb(0.47408991209053464,0.8205325891656352,0.32029128104313065)  );
  draw(  ellipsoid( (-0.7108655269641794,-0.4665710391531471,0.526290478725857), (0.5836143440917645,-0.8089055776121944,0.0711762873309745), (0.3925105093424619,0.3577474415442188,0.8473206406811338), 0.061138468483848685, 1.1394944162252392, 16.0686859074294,  (gDx*2, gDy*3, gDz*1)), rgb(0.4717762073601761,0.8199149320886387,0.32171191680128003)  );
  draw(  ellipsoid( (-0.7888751631365274,-0.41341982609602534,0.4547087247646013), (0.5177144791342296,-0.8457349709587056,0.12924425322717675), (0.33113093344037015,0.33736687194834325,0.8812127431163824), 0.06399770717006455, 1.28276815800682, 13.322390425426752,  (gDx*2, gDy*3, gDz*2)), rgb(0.4558902890428174,0.8155867556486258,0.33141918277291876)  );
  draw(  ellipsoid( (-0.8500704847730753,-0.376934583935938,0.3678321496969916), (0.4551799470632478,-0.877140515612878,0.1530873334468706), (-0.2649365711055546,-0.2975648421830356,-0.9172043272834122), 0.06606629031711955, 1.407677595364148, 10.375816208320789,  (gDx*2, gDy*3, gDz*3)), rgb(0.43634487763726515,0.8100340010818563,0.3432491501088384)  );
  draw(  ellipsoid( (-0.4152986215214449,-0.6135636007794064,0.6716150406006378), (0.7119623979569164,-0.6787901357730067,-0.17987077437065918), (-0.5662478245943423,-0.40346457016183235,-0.7187348201988704), 0.06726255810935561, 0.47390729404811827, 2.271543950095764,  (gDx*3, gDy*0, gDz*0)), rgb(0.2890084641818351,0.7584306497835237,0.42837736121245656)  );
  draw(  ellipsoid( (-0.46850490807861556,-0.6113368647796877,0.6377855351664771), (0.6865669848734071,-0.7062717388431956,-0.17264416061149532), (-0.5559936388018182,-0.35699785527752204,-0.7506154840790075), 0.04404182468234395, 0.2504081191276107, 1.4374550354163753,  (gDx*3, gDy*0, gDz*1)), rgb(0.28444552789099653,0.7564755110827741,0.4309066284150656)  );
  draw(  ellipsoid( (-0.557011644847852,-0.5862693634570558,0.5882399688694645), (0.6457651957855787,-0.7511222275981474,-0.13712297808905047), (-0.5222311168591005,-0.30348580309407436,-0.7969761777521592), 0.03335190894653214, 0.14438555389442637, 0.9146932358738944,  (gDx*3, gDy*0, gDz*2)), rgb(0.2655664476003493,0.7481186360431309,0.44132857694018063)  );
  draw(  ellipsoid( (-0.6859618066124504,-0.5136403113125376,0.5153930834457745), (0.5559535998887071,-0.8269395796314847,-0.0841815087159324), (0.4694379561132341,0.2287893402972765,0.8528091481255149), 0.03004914867677796, 0.09842661231690317, 0.6032753248604592,  (gDx*3, gDy*0, gDz*3)), rgb(0.23086183177666017,0.7312247668958868,0.4603457950764054)  );
  draw(  ellipsoid( (-0.4255927544798215,-0.6454354974634594,0.6342584851213068), (-0.7473582654451302,0.6458972889789492,0.15579574821056605), (-0.5102219423007258,-0.407712779666936,-0.7572607601686062), 0.057549413253839325, 0.5643940791781278, 3.6593594948944403,  (gDx*3, gDy*1, gDz*0)), rgb(0.3548438393366551,0.7838861673500946,0.39120692551230746)  );
  draw(  ellipsoid( (-0.5078598075145137,-0.6227425162103063,0.5952059932623138), (-0.7083232333054029,0.6951087912881802,0.1228900542503962), (-0.49026178014516,-0.3591873143332616,-0.7941208095440879), 0.04432389027057306, 0.33080745627582553, 2.4016532322058306,  (gDx*3, gDy*1, gDz*1)), rgb(0.33833874130108077,0.7779202121208395,0.4006502865559354)  );
  draw(  ellipsoid( (-0.6251809418617977,-0.5729093146907293,0.5300223646917124), (-0.6366907001786987,0.7671422682525684,0.07821567979799647), (-0.4514130505853296,-0.28856135812146344,-0.8443687585174756), 0.03802696405962953, 0.2220446258508461, 1.6513243410482568,  (gDx*3, gDy*1, gDz*2)), rgb(0.3157608008034718,0.7693263500868004,0.41342872694538385)  );
  draw(  ellipsoid( (-0.7608574953228627,-0.48416864633817286,0.4320608680661574), (0.5212533808173544,-0.8525786719513637,-0.03747694117890573), (-0.38651104097748384,-0.19669857660062073,-0.9010654166961417), 0.03548344492972377, 0.17734413476129676, 1.1618416720100382,  (gDx*3, gDy*1, gDz*3)), rgb(0.28519357165027137,0.7567961730435271,0.43049200878170074)  );
  draw(  ellipsoid( (-0.44787259269198987,-0.6634871251039495,0.5993287708233698), (0.767860131455351,-0.6288290355131764,-0.12233095526864635), (-0.4580403467345711,-0.40541198664675027,-0.791100601596541), 0.060966458399452264, 0.7114280726641932, 5.938694037394824,  (gDx*3, gDy*2, gDz*0)), rgb(0.3940204519407423,0.7971013934840684,0.36843387312617926)  );
  draw(  ellipsoid( (-0.5624575133001055,-0.6269465215887924,0.5390543616370947), (-0.712719135871562,0.6981180516352345,0.06828337530847332), (-0.41913360530355276,-0.34578786133804523,-0.8394985264171144), 0.053651544684819516, 0.49123918567043867, 4.335397286353513,  (gDx*3, gDy*2, gDz*1)), rgb(0.3768461049994164,0.7914658960610989,0.3784808742122479)  );
  draw(  ellipsoid( (-0.6929088707836886,-0.5554341395893786,0.4597501640759689), (0.6136513799035045,-0.7890652037737714,-0.028426891071328127), (-0.3785621226850335,-0.2624290776039563,-0.887593205526123), 0.04989362026155322, 0.38350821549972514, 3.1839625323141125,  (gDx*3, gDy*2, gDz*2)), rgb(0.3545478552169552,0.7837818789471028,0.39137721968068745)  );
  draw(  ellipsoid( (-0.8160107431098526,-0.45660688546519856,0.3544525627994205), (0.4830739812062461,-0.8754401059996708,-0.0156252836400663), (0.31743660124638046,0.15847641134584012,0.9349434374528175), 0.046170921153628405, 0.3459152898165559, 2.4667820379727114,  (gDx*3, gDy*2, gDz*3)), rgb(0.3369208085433873,0.7773988547471168,0.4014585549785649)  );
  draw(  ellipsoid( (-0.4767043827126108,-0.66817001767043,0.5712282897308973), (-0.7742130715502215,0.6268921304772382,0.08718013871565561), (-0.4163496743644215,-0.4006932545376915,-0.8161481877849346), 0.07484013728534515, 0.9447420995854778, 8.848502728000847,  (gDx*3, gDy*3, gDz*0)), rgb(0.4106006967495613,0.8023233349651353,0.35863972464902016)  );
  draw(  ellipsoid( (-0.6148649400965586,-0.6120034528793513,0.49738604635012346), (-0.6965222756179007,0.7172143870832831,0.02145326382655733), (-0.36986189991416624,-0.33324960108678653,-0.8672640188358912), 0.07062219504354579, 0.7487892914229672, 7.498276221034618,  (gDx*3, gDy*3, gDz*1)), rgb(0.4011856390365317,0.7993819433898595,0.3642121284484935)  );
  draw(  ellipsoid( (-0.7505764429381924,-0.5212533561127523,0.40611567569775336), (0.5759100221143234,-0.8173701793190712,0.01528516889630482), (-0.32397939708340917,-0.24535877547196458,-0.9136938336031214), 0.06674328283638536, 0.6642729422450604, 6.028009319120484,  (gDx*3, gDy*3, gDz*2)), rgb(0.3869710206146626,0.7948182519574545,0.372570302970758)  );
  draw(  ellipsoid( (0.8527300979728017,0.42539146043780424,-0.30313938278931674), (-0.4483242921257094,0.8938450082869749,-0.006813974643279985), (-0.26806101749653416,-0.14171523046959314,-0.9529218668661523), 0.05954052474699352, 0.6630786597709619, 4.865840818896169,  (gDx*3, gDy*3, gDz*3)), rgb(0.37847995544221186,0.792012896860533,0.37752971783574113)  );
