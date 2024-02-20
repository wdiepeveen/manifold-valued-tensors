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

  draw(  ellipsoid( (0.08176632988932775,-0.6577673578679848,0.7487699047236075), (-0.4292515778344357,0.6547919039621869,0.6220857219324534), (-0.8994761532264639,-0.37227632939651845,-0.2288077453845872), 0.953773581313969, 3.535636769841778, 47.63189712164687,  (gDx*0, gDy*0, gDz*0)), rgb(0.3371987687282492,0.7775010574723115,0.401300108508552)  );
  draw(  ellipsoid( (0.004135232916761988,-0.6249937577875289,0.780618794659306), (0.4976334162685132,-0.6758166740832808,-0.5437212576715584), (0.8673775895546603,0.39071041163198594,0.308223119477938), 0.6718411725762836, 3.078852202282413, 60.86228624563481,  (gDx*0, gDy*0, gDz*1)), rgb(0.3941685795786281,0.7971493684689505,0.36834695553611907)  );
  draw(  ellipsoid( (-0.0564162663732799,-0.5978726671226374,0.7996033259036401), (0.5354849442213232,-0.6940632429339393,-0.4811778146593107), (-0.8426583408480482,-0.4010292866059628,-0.3593082685988828), 0.466004543525878, 2.4735007758562544, 73.78532434654299,  (gDx*0, gDy*0, gDz*2)), rgb(0.4416243231269541,0.8115617525537742,0.34006745227843305)  );
  draw(  ellipsoid( (-0.10531727417533493,-0.5763286181785169,0.810403353663297), (0.556666589023429,-0.7094564932535876,-0.432196474818274), (0.8240331185245515,0.40560671598084563,0.3955409100534808), 0.3210400425791033, 1.8921752075428915, 80.05398477562991,  (gDx*0, gDy*0, gDz*3)), rgb(0.47600938710923624,0.8210450043039469,0.3191127059152578)  );
  draw(  ellipsoid( (0.12539056963332681,-0.6577140912735432,0.7427579546442078), (-0.40549530808766576,0.6493019039511697,0.6434132362986287), (-0.9054561061508827,-0.3818628178621251,-0.1852836424741238), 0.4293682732087896, 1.6519805679699389, 38.939074820773335,  (gDx*0, gDy*1, gDz*0)), rgb(0.39790348673840986,0.7983408165704711,0.3661475088395352)  );
  draw(  ellipsoid( (0.05340682533035117,-0.6327505057599495,0.7725118176886755), (0.47930091988046797,-0.6624313565879307,-0.5757215698675975), (0.876024165882352,0.4010130861575488,0.2678995437109359), 0.31868885913679584, 1.5079235080689801, 49.386267327939144,  (gDx*0, gDy*1, gDz*1)), rgb(0.44189879345272143,0.8116399517691103,0.33990144991370996)  );
  draw(  ellipsoid( (-0.004153293661477065,-0.61558660933861,0.7880582951500195), (0.5257270482843128,-0.6717121343211957,-0.5219328302645154), (0.8506431806897798,0.4121358210695332,0.3264203488407493), 0.24340435798662075, 1.3405894491890085, 60.366750446096646,  (gDx*0, gDy*1, gDz*2)), rgb(0.4767187464087373,0.8212343719383391,0.3186771528505023)  );
  draw(  ellipsoid( (-0.05863511869956006,-0.5945495576517191,0.8019181668669401), (0.5548375344292231,-0.6872089181280837,-0.4689341246206437), (-0.8298898922408859,-0.4174382904590498,-0.3701729871493015), 0.2007158444026015, 1.177951147809339, 66.08313228537767,  (gDx*0, gDy*1, gDz*3)), rgb(0.4963613562253832,0.826311273092451,0.30653422792745044)  );
  draw(  ellipsoid( (0.1641878798582569,-0.6434524879426278,0.747670539655063), (-0.39269528919793195,0.6526697502757015,0.6479294768080908), (-0.9048937782686998,-0.3999888658894089,-0.1455202982923736), 0.1697574930416405, 0.6035866190250817, 27.134464614322276,  (gDx*0, gDy*2, gDz*0)), rgb(0.4506720741805923,0.8141316441541626,0.3345912697571878)  );
  draw(  ellipsoid( (0.10107605476912818,-0.6318363189224458,0.7684832446077412), (-0.4577229337469627,0.6562981236370033,0.5998020413708359), (-0.8833308253844546,-0.4123780292414734,-0.22286994845545377), 0.13561806199839532, 0.5815800204508337, 32.6732717734928,  (gDx*0, gDy*2, gDz*1)), rgb(0.479227690899006,0.8218938395173314,0.3171316771328753)  );
  draw(  ellipsoid( (0.04280458524923901,-0.6170418817465229,0.78576528534439), (0.5073931269250139,-0.664084983118911,-0.5491296294548423), (0.8606509061349424,0.42219717119689865,0.28465692755040994), 0.1208085485787248, 0.5960153054624364, 37.91292990363723,  (gDx*0, gDy*2, gDz*2)), rgb(0.4959666094009435,0.82621053834645,0.3067789223819785)  );
  draw(  ellipsoid( (-0.01431211944801651,-0.5944300488204965,0.8040199501854216), (0.542025077904744,-0.680320729933971,-0.49332800381132325), (0.840240428781704,0.4287384068186222,0.3319328250706274), 0.11713461478262573, 0.5829989785827989, 39.89243926775504,  (gDx*0, gDy*2, gDz*3)), rgb(0.5015958627126241,0.8276191523547457,0.30327427742166174)  );
  draw(  ellipsoid( (0.16865252569838057,-0.571904607365212,0.8027960174602263), (0.3996561757743783,-0.7048326255847793,-0.5860767109147621), (-0.9010167960370975,-0.4196857038138294,-0.10969340578783401), 0.06423071159811193, 0.2185738985929892, 18.83518153999195,  (gDx*0, gDy*3, gDz*0)), rgb(0.49879629461086256,0.8269204200509713,0.30501820887693265)  );
  draw(  ellipsoid( (0.12175173117023963,-0.5777977003482869,0.807047912722213), (0.44246385628949103,-0.6962428003363684,-0.5652182754097788), (-0.8884831184859295,-0.4259058351936529,-0.17088583239553462), 0.05905795533212265, 0.2260032147097497, 20.80933399382044,  (gDx*0, gDy*3, gDz*1)), rgb(0.5091478687283363,0.8294879784614355,0.29856177445998666)  );
  draw(  ellipsoid( (0.07031181037392836,-0.5717210369501602,0.817429694365561), (-0.4821926992407985,0.6978781312039112,0.5295812636283588), (-0.8732390566854035,-0.43139444815153344,-0.22661063519442512), 0.05883853485290565, 0.24339380403788557, 21.765868684100308,  (gDx*0, gDy*3, gDz*2)), rgb(0.5108841755909074,0.8299116509105681,0.29747526539250185)  );
  draw(  ellipsoid( (0.013260481270337021,-0.5537822974207958,0.8325559000450508), (0.5166729772640455,-0.7090651013842003,-0.4798705206241795), (-0.8560801330492295,-0.43652244966594755,-0.27672180386817063), 0.061718182590543506, 0.2592575524545445, 21.633134256861354,  (gDx*0, gDy*3, gDz*3)), rgb(0.5068561844404962,0.8289287895785266,0.2999958157646808)  );
  draw(  ellipsoid( (0.048109915466172715,-0.6745502373032044,0.7366596319794023), (0.49059308941026564,-0.6264668462813885,-0.6056878000530821), (-0.8700596856864067,-0.3905397135560501,-0.30079041786404936), 1.699024539540519, 5.944267870916132, 52.38762693412363,  (gDx*1, gDy*0, gDz*0)), rgb(0.2826915885699055,0.7557236545038377,0.43187878768949084)  );
  draw(  ellipsoid( (-0.01407912486484253,-0.6625809186048794,0.7488580002535556), (0.5470640764301842,-0.6320020186608207,-0.5489028554200072), (0.836972326019582,0.4019452384459392,0.37137225363239973), 1.1256266366940508, 4.938786767406153, 64.24964007008414,  (gDx*1, gDy*0, gDz*1)), rgb(0.34834742991291334,0.7815669396572903,0.39493380896249414)  );
  draw(  ellipsoid( (-0.06207933653835984,-0.6475816744449557,0.7594630543337999), (0.578629979285465,-0.6433548377671986,-0.5012802607261171), (0.813224140718326,0.4083289453915892,0.4146492123570633), 0.7512941135005284, 3.900368228349218, 74.11995637491131,  (gDx*1, gDy*0, gDz*2)), rgb(0.4000771087894279,0.7990303081013252,0.3648657933330721)  );
  draw(  ellipsoid( (-0.09239147293355708,-0.6381633868959534,0.7643371686334857), (0.5942935085413292,-0.6512462728656833,-0.4719041404609554), (0.7989236768520536,0.41064069903510786,0.4394259606135638), 0.48437069364499713, 2.7867797318160865, 76.7808913181965,  (gDx*1, gDy*0, gDz*3)), rgb(0.44039579437375836,0.8112117329869987,0.34081047860592645)  );
  draw(  ellipsoid( (0.06907529012998337,-0.6570030120677334,0.7507167551263155), (-0.4724815224663503,0.6412189289328897,0.6046482416298237), (-0.8786295096463232,-0.39646604812024444,-0.26612939985386397), 0.887160367868941, 3.8720052550871937, 53.15231291191973,  (gDx*1, gDy*1, gDz*0)), rgb(0.3535514142035376,0.783430788347816,0.391950521025002)  );
  draw(  ellipsoid( (0.01428493427759676,-0.6486536386623375,0.760949668311114), (0.5307649856800881,-0.6400402723451127,-0.5555510595367537), (0.8473986491780008,0.41182145017774796,0.33513970601048754), 0.6242960402860778, 3.338409725418555, 66.6197046493558,  (gDx*1, gDy*1, gDz*1)), rgb(0.4068713569724297,0.8011658670764968,0.3608504632085563)  );
  draw(  ellipsoid( (-0.025610037529505837,-0.6424339248319998,0.7659130356657274), (0.567207660181779,-0.6402336758252662,-0.5180504903678591), (0.8231765279749227,0.421164448362181,0.380789851790894), 0.4282142786583794, 2.746503246077813, 78.80875547615089,  (gDx*1, gDy*1, gDz*2)), rgb(0.45104706483185514,0.8142362111797404,0.334363317670986)  );
  draw(  ellipsoid( (-0.058962078629798784,-0.6354983232864055,0.7698476176385953), (0.5916435713094775,-0.6433946164888593,-0.48579960065990246), (0.8040406443780195,0.4268316396113575,0.413924381517759), 0.30046553333112247, 2.178749285940522, 83.15308611373194,  (gDx*1, gDy*1, gDz*3)), rgb(0.48073865693684215,0.8222881632522634,0.3161995815115511)  );
  draw(  ellipsoid( (0.10411988078494085,-0.6499292386178585,0.7528286891550697), (0.4525003787277819,-0.6431049226393032,-0.6177859384352397), (0.8856649804957535,0.4049790652202989,0.2271332187434272), 0.40877223677346813, 1.8580538211079625, 41.20783281988526,  (gDx*1, gDy*2, gDz*0)), rgb(0.40443446385840637,0.8004095333294828,0.36229504425898107)  );
  draw(  ellipsoid( (0.053716456107963384,-0.6449770229410074,0.7623117355920452), (-0.5153787962501909,0.6359672181168862,0.574395677087122), (-0.8552772876201644,-0.4237338048336763,-0.2982455765342889), 0.2950287684211033, 1.7112910765864562, 52.17345881811636,  (gDx*1, gDy*2, gDz*1)), rgb(0.4493060630448457,0.8137503535703547,0.33542146008239915)  );
  draw(  ellipsoid( (0.016372502405421152,-0.644437160681537,0.7644819730364498), (0.5567014030869422,-0.6292363975876,-0.5423514577762373), (0.8305513162749273,0.43446783757109053,0.34845689711969546), 0.22390247842296707, 1.55208762537709, 61.88754501632561,  (gDx*1, gDy*2, gDz*2)), rgb(0.48125450495381467,0.8224227864745042,0.3158813614683147)  );
  draw(  ellipsoid( (-0.018371902733801456,-0.6411808770509797,0.7671698352347257), (0.584754689804257,-0.6292769968752913,-0.5119300869801869), (0.8110021121685606,0.43920102926535876,0.3864945406475788), 0.1860680504728145, 1.3964224639992482, 64.7893370668996,  (gDx*1, gDy*2, gDz*3)), rgb(0.49668330000893046,0.8263930467089398,0.30633445396088266)  );
  draw(  ellipsoid( (0.14031798919981878,-0.6274755301288989,0.7658885826256825), (-0.4332254574050108,0.6566524756723573,0.617351787273337), (-0.8902957738657076,-0.41842799294839533,-0.17969821856111573), 0.15789421841760598, 0.6813381206881018, 27.324536262903862,  (gDx*1, gDy*3, gDz*0)), rgb(0.4529463561382645,0.8147658330301737,0.3332087624463613)  );
  draw(  ellipsoid( (0.0937309866476113,-0.6324938340978081,0.7688732353062628), (0.49190417299152156,-0.6420065907071084,-0.5880967795203119), (0.8655892713842934,0.43333484433736447,0.25095044520287263), 0.126722175597084, 0.7012809470454999, 33.15434434251823,  (gDx*1, gDy*3, gDz*1)), rgb(0.48065471702872975,0.8222662570695645,0.3162513629671541)  );
  draw(  ellipsoid( (0.05538755836070831,-0.6339275944059375,0.7714065228072291), (0.534885529643869,-0.6335411899750638,-0.5590375933535936), (0.8431071631509465,0.4435779138501821,0.3039883974560273), 0.11312013420979748, 0.7205525329124288, 37.56733643194921,  (gDx*1, gDy*3, gDz*2)), rgb(0.49568725451628637,0.8261392502660707,0.30695208803370466)  );
  draw(  ellipsoid( (0.02226398509013372,-0.6344228964973658,0.7726654537170656), (0.5667929495630176,-0.6286416704705803,-0.5324992041999808), (0.8235593890506988,0.44979688588055944,0.3455903559976615), 0.10398151010934825, 0.7129729498986287, 38.45949103201673,  (gDx*1, gDy*3, gDz*3)), rgb(0.5021055792479612,0.8277463703383996,0.3029567603379778)  );
  draw(  ellipsoid( (0.053160696199400115,-0.7086187952602124,0.7035860596853523), (0.49514756116967024,-0.5931602075111918,-0.6348148240967656), (-0.8671809690297192,-0.38212611953286413,-0.31933805868300846), 2.79281242177227, 7.0392380326629125, 43.59928674633785,  (gDx*2, gDy*0, gDz*0)), rgb(0.20429538081770268,0.7164972469151442,0.4749366486309339)  );
  draw(  ellipsoid( (0.0007418273128375236,-0.7135444534565414,0.7006095650457846), (0.5478523366388124,-0.5858225821905828,-0.5972183180671463), (0.8365747228619996,0.3842736201419041,0.3904824169298845), 1.7748389095108419, 5.731944342868402, 51.20721194443219,  (gDx*2, gDy*0, gDz*1)), rgb(0.27631124598502255,0.7529366382153175,0.43540627647470576)  );
  draw(  ellipsoid( (-0.03303605131033774,-0.7205172382509769,0.6926496435406617), (0.5769901383868086,-0.5796261170690478,-0.5754267500002325), (0.8160827161006367,0.3806421860417515,0.4348798991528171), 1.1380780837511701, 4.39713015859055, 55.58454689879415,  (gDx*2, gDy*0, gDz*2)), rgb(0.3338722835211913,0.7762620805213721,0.40319133914627403)  );
  draw(  ellipsoid( (-0.046682999536848034,-0.7273218799078632,0.6847069304173389), (0.5919541416649529,-0.5722740020543574,-0.5675321671398001), (0.8046185380165881,0.3788209993915067,0.4572566661078528), 0.7149407921213454, 3.12659889546052, 54.37713141248881,  (gDx*2, gDy*0, gDz*3)), rgb(0.3779550165786389,0.7918392315073541,0.37783622535568956)  );
  draw(  ellipsoid( (0.061136518168992536,-0.6984651085153646,0.7130279225477726), (0.48547624395622435,-0.603362992651256,-0.6326657218888002), (0.8721095932853445,0.38483709707555586,0.3022007048537945), 1.6413932267949567, 5.809596867685204, 53.74923819827877,  (gDx*2, gDy*1, gDz*0)), rgb(0.28967566364179853,0.758710223836144,0.42800633192822446)  );
  draw(  ellipsoid( (0.013719533911323745,-0.6988191303854832,0.7151668318627012), (0.5426213653958016,-0.5955535818277748,-0.5923495462884255), (-0.8398653631417667,-0.39619156247878995,-0.37102347313258105), 1.095848370240521, 4.972041561840824, 64.18568645703795,  (gDx*2, gDy*1, gDz*1)), rgb(0.35050176877082706,0.7823422520485888,0.3937001210695662)  );
  draw(  ellipsoid( (-0.01863202041643708,-0.7002751887435612,0.7136298114886817), (0.5772847136874083,-0.5902947277124082,-0.5641759422181886), (-0.8163303296815869,-0.4014558437301706,-0.41525654525467526), 0.7338270838297398, 3.9878861196738438, 71.88949524135411,  (gDx*2, gDy*1, gDz*2)), rgb(0.3987098213846435,0.7985965928262152,0.3656720390498392)  );
  draw(  ellipsoid( (-0.03462242253639303,-0.7068145889896662,0.7065510771692878), (0.5981787597055784,-0.5810098600272104,-0.5519145893960924), (-0.800614426109602,-0.40353522689169374,-0.4429174430553194), 0.48385966288897936, 2.9761473191336254, 71.79738785976325,  (gDx*2, gDy*1, gDz*3)), rgb(0.43384655615330103,0.8093066658453734,0.34475267349262234)  );
  draw(  ellipsoid( (0.07562700551964165,-0.6768101122641672,0.7322626768947722), (0.47268848286081716,-0.62227529918929,-0.6239703920777159), (0.87797844746148,0.393321146082745,0.2728595276644833), 0.852724026444224, 3.693839408096586, 51.11522308648348,  (gDx*2, gDy*2, gDz*0)), rgb(0.35374151653496005,0.7834977698756398,0.3918411458380172)  );
  draw(  ellipsoid( (0.034461049358837224,-0.6828724540601243,0.7297243641012621), (0.5326308331824461,-0.6052730136764728,-0.5915648522844928), (0.845645807427945,0.409059641618315,0.34286058096350863), 0.5909191431080846, 3.3550423649035093, 61.75056356022852,  (gDx*2, gDy*2, gDz*1)), rgb(0.4040386942973622,0.8002866991053078,0.3625296549626386)  );
  draw(  ellipsoid( (0.008268698527689536,-0.6862956161237326,0.7272757083211321), (0.5722022915931441,-0.5932110350876945,-0.5662907427688895), (0.8200708299410562,0.420831314352356,0.3877948410438897), 0.4094989581729065, 2.8556809928789706, 69.88870203972459,  (gDx*2, gDy*2, gDz*2)), rgb(0.443825929079881,0.812189011101732,0.3387358992467798)  );
  draw(  ellipsoid( (-0.007352504309727913,-0.6911248588624623,0.7226979798938953), (0.5995993215195955,-0.5814293333263029,-0.5499277988799727), (0.8002665770314389,0.42928587189665784,0.4186729581383426), 0.294321041616678, 2.363795654699009, 71.1157046848926,  (gDx*2, gDy*2, gDz*3)), rgb(0.4694128920893106,0.8192840311862586,0.3231630138827353)  );
  draw(  ellipsoid( (0.10192084586770293,-0.6599135378779153,0.744396576901765), (0.45589483351762966,-0.6341035991357967,-0.624557864681057), (0.8841787386889287,0.4030220193593432,0.23622279729520682), 0.3874005741825985, 1.7600965732342582, 37.08142687642657,  (gDx*2, gDy*3, gDz*0)), rgb(0.399579249293395,0.7988723828052007,0.36515936517818953)  );
  draw(  ellipsoid( (0.0651638966328784,-0.671024925334239,0.738565647830847), (0.5171566302817558,-0.610274667613502,-0.600094867354195), (0.8534065188005296,0.42105864158025225,0.3072571138628914), 0.28347429204763597, 1.7310087228455233, 45.43930979427219,  (gDx*2, gDy*3, gDz*1)), rgb(0.4404279113625817,0.8112208834236367,0.3407910539337782)  );
  draw(  ellipsoid( (0.042025043149462635,-0.6776288565660373,0.7342023076082577), (0.5611990944690767,-0.591965350342449,-0.5784743731238488), (0.8266132542582258,0.4363440806782487,0.35540733129984764), 0.21379364321155442, 1.6532407417533832, 50.82562733504266,  (gDx*2, gDy*3, gDz*2)), rgb(0.4686190347327873,0.8190721062988914,0.32365044952339805)  );
  draw(  ellipsoid( (0.02677539946865972,-0.6840681258393734,0.7289265238649921), (0.5920337230650758,-0.5766968329704342,-0.56295366913768), (0.8054682791624997,0.44662239313818425,0.3895501112957386), 0.17168880349652685, 1.5387338000600388, 51.29705299581509,  (gDx*2, gDy*3, gDz*3)), rgb(0.48381408144818794,0.8230907708883373,0.31430239146958033)  );
  draw(  ellipsoid( (0.09887602004361408,-0.7581109730163824,0.6445861348590184), (0.45729317640009837,-0.5406945458067547,-0.7060682395864695), (0.883802287549791,0.36457805848265384,0.2932172501657691), 4.036019937588669, 6.42607136719738, 27.039961192005972,  (gDx*3, gDy*0, gDz*0)), rgb(0.12609383218304193,0.6435594683954533,0.5255748543689491)  );
  draw(  ellipsoid( (0.06058185398905636,-0.7881385400809481,0.6125091677732837), (0.5000560749626937,-0.5071142063272216,-0.7019822673216304), (0.8638713798130793,0.3488163175371599,0.36339154607621454), 2.53114253653626, 5.247257540053019, 30.1285187115905,  (gDx*3, gDy*0, gDz*1)), rgb(0.17537746188985265,0.6976565337265648,0.49122480932235935)  );
  draw(  ellipsoid( (0.04078283608335409,-0.8042872164001762,0.5928396358343921), (-0.5278635318133592,0.4864297384535921,0.6962371731884879), (0.8483494869819425,0.34133295047977186,0.40471590635191085), 1.5715357611799492, 4.0583281163639855, 30.778986831757784,  (gDx*3, gDy*0, gDz*2)), rgb(0.23295473459118643,0.7323181775071129,0.4592005390383034)  );
  draw(  ellipsoid( (0.030438821170678053,-0.8113952075570849,0.5837048015213979), (0.5458938209067344,-0.4756950742609801,-0.6897203292782421), (0.8373012686372723,0.33963511814480457,0.4284560328215475), 0.9961434715622108, 2.9866372837278745, 29.515556145756445,  (gDx*3, gDy*0, gDz*3)), rgb(0.2813605034805484,0.7551518896956828,0.43261636877828324)  );
  draw(  ellipsoid( (0.07781630134101858,-0.7368665731633044,0.6715446944174097), (0.4734821253413718,-0.5654604688882573,-0.6753289088339766), (-0.8773592765952961,-0.3705160070486032,-0.30489110891093263), 2.7928425985226886, 6.424865256883602, 37.827737355045834,  (gDx*3, gDy*1, gDz*0)), rgb(0.18857771490792263,0.7067209668937312,0.4836989651668022)  );
  draw(  ellipsoid( (0.04208133931554576,-0.753019368396597,0.6566513471401697), (0.5202027048590737,-0.5446073770036388,-0.6578692505129874), (0.8530054552807462,0.3692758260877719,0.3688049044257416), 1.7683370262948936, 5.440986210996072, 43.273769890443205,  (gDx*3, gDy*1, gDz*1)), rgb(0.2569764708235771,0.7441337050549234,0.4460471925648759)  );
  draw(  ellipsoid( (0.020331649110274852,-0.7712566893154548,0.6361994523972972), (-0.5503067538805096,0.5226319413520664,0.6511668991219275), (-0.8347149817021094,-0.3633441523744875,-0.4138017958604247), 1.138399517434146, 4.36241776078167, 44.96153581606297,  (gDx*3, gDy*1, gDz*2)), rgb(0.3100943347163302,0.7670880390440423,0.41661161510106076)  );
  draw(  ellipsoid( (0.011546905847514409,-0.7838896689391275,0.6207927640492067), (0.5699574507113376,-0.5049473882802711,-0.6482103358074186), (0.8215930704089354,0.36131028493448464,0.44095317739707895), 0.7338752303113333, 3.2570466987626716, 42.00440554941633,  (gDx*3, gDy*1, gDz*3)), rgb(0.3484414869875022,0.7816007893068708,0.39487994692786654)  );
  draw(  ellipsoid( (0.07549491017997878,-0.709411778448002,0.7007392147840449), (-0.47037201429083636,0.5943085963468675,0.6523399884111025), (-0.879233010474981,-0.3788564647824548,-0.28882017308983526), 1.6359761796636598, 5.094665899609623, 41.90197715619251,  (gDx*3, gDy*2, gDz*0)), rgb(0.2624058812708776,0.7466646149155548,0.4430659068790518)  );
  draw(  ellipsoid( (0.04345155499689876,-0.7253133006437058,0.6870462708418441), (0.5221589367017012,-0.5698046421906281,-0.6345649805658669), (0.8517405750525804,0.38632018539981,0.35396992409208444), 1.0799513266833975, 4.586113345904754, 48.19159570662261,  (gDx*3, gDy*2, gDz*1)), rgb(0.3221078148519433,0.7717964547810264,0.40985335813046603)  );
  draw(  ellipsoid( (0.02185048836666803,-0.7416910683497364,0.6703856466902947), (-0.5577662354125865,0.5474479991863566,0.6238569666362044), (-0.8297104210456858,-0.38755005782102714,-0.4017282288949947), 0.720235259702538, 3.8282323393970032, 51.64467152936065,  (gDx*3, gDy*2, gDz*2)), rgb(0.36886758810796977,0.7887684545766203,0.3831143326178388)  );
  draw(  ellipsoid( (0.014391556339834728,-0.7577490816308993,0.6523873177749904), (0.5824611594012788,-0.5239721531401672,-0.621443626182275), (0.8127311246008948,0.3889338144481448,0.4338186338595668), 0.47810342855279464, 3.007626430136655, 49.283607705002034,  (gDx*3, gDy*2, gDz*3)), rgb(0.4014573445224596,0.799468130697053,0.3640519124006568)  );
  draw(  ellipsoid( (0.08068382866602675,-0.6786749280363439,0.7299934669890237), (0.4581810496277397,-0.6251630842691867,-0.6318553978791829), (0.8851893840042446,0.3854496855906117,0.26051543970422264), 0.8314052129809081, 3.1288107428137173, 36.19992539709124,  (gDx*3, gDy*3, gDz*0)), rgb(0.32157180525131873,0.771590694741523,0.4101561595976024)  );
  draw(  ellipsoid( (0.05381371766320319,-0.70177547049209,0.7103627754935294), (0.5126001315591278,-0.5910912041099986,-0.622777884602012), (0.8569394313000364,0.39764604542155024,0.327921383328036), 0.5723983213380903, 3.018513752823966, 41.82594238825524,  (gDx*3, gDy*3, gDz*1)), rgb(0.37086302674333077,0.7894449917534683,0.381956159013652)  );
  draw(  ellipsoid( (0.03617479751075238,-0.7179705179300927,0.6951328789578655), (0.5551431081766375,-0.5639526285973047,-0.6113702332810887), (0.8309678173485651,0.40801442141348854,0.37817551275773786), 0.3961581562420636, 2.7098050459261027, 45.24167127807497,  (gDx*3, gDy*3, gDz*2)), rgb(0.40977134541318705,0.8020659308138325,0.35913136099440446)  );
  draw(  ellipsoid( (0.028741392784643664,-0.7326283750258488,0.6800217617457421), (0.585642170998776,-0.5389719190615826,-0.6054192910792614), (0.8100599853644417,0.4156500145194985,0.41356726845980124), 0.281941800328561, 2.3344250275390057, 44.50545639123698,  (gDx*3, gDy*3, gDz*3)), rgb(0.4356377504839272,0.8098281354642415,0.34367470871575756)  );
