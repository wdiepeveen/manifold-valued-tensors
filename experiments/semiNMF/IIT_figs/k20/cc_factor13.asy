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

  draw(  ellipsoid( (0.13407790723221907,-0.12472033349726643,0.9830910197965187), (0.4859346755977691,-0.8563140946406177,-0.1749104410019886), (0.8636495851002995,0.5011696416705125,-0.05420686694442669), 0.13223285705696222, 0.25157640349293126, 1.0000000000000004,  (gDx*0, gDy*0, gDz*0)), rgb(0.13099414834260598,0.6529130997199726,0.5207878465138328)  );
  draw(  ellipsoid( (0.06991774375456365,-0.04958368182514495,0.9963197115408958), (0.4858163440753264,-0.8706253387028984,-0.07742092377224333), (-0.8712600007675981,-0.4894414961005312,0.03678359630757331), 0.12269820740396771, 0.23956548247969756, 0.9978491925794254,  (gDx*0, gDy*0, gDz*1)), rgb(0.1361894684208729,0.6606816538760394,0.5164734278887141)  );
  draw(  ellipsoid( (0.02088394501163941,-0.03251341603882495,0.9992530903721225), (0.48780789629193366,-0.8720984631621449,-0.038571062539983485), (0.8727011614269738,0.48824906382676403,-0.0023525552702498564), 0.12110132444501281, 0.2428429880535328, 0.989396687861885,  (gDx*0, gDy*0, gDz*2)), rgb(0.1361087741505717,0.6605714194667425,0.5165367760918443)  );
  draw(  ellipsoid( (-0.014634933073554663,-0.040145504277665965,0.9990866615165196), (0.49317911497105826,-0.8694862363243183,-0.027713632005480892), (-0.8698046788160525,-0.49232308835648086,-0.032523797146709485), 0.12809339215395998, 0.2482874128464662, 0.9723626918186581,  (gDx*0, gDy*0, gDz*3)), rgb(0.13098173089514648,0.6528926202405188,0.5207988309619036)  );
  draw(  ellipsoid( (0.12266925010556225,-0.16576588803270553,0.9785059659722405), (0.5207803594376751,-0.8285506265558976,-0.20564940179294394), (0.844831386906586,0.5348135465735304,-0.015310065281927908), 0.16983038312510768, 0.27037817797180874, 0.9462067363209349,  (gDx*0, gDy*1, gDz*0)), rgb(0.11969655874003512,0.6184484872784677,0.536362201364133)  );
  draw(  ellipsoid( (0.03996827567127413,-0.04393276865040516,0.9982346661875527), (0.5154522607720927,-0.8549352269233575,-0.05826426546731279), (0.8559856913551785,0.5168710376914317,-0.011525041907927405), 0.1647042193751364, 0.2489472705352581, 0.9553493836220843,  (gDx*0, gDy*1, gDz*1)), rgb(0.12073262944049558,0.626295280687299,0.5332947875979099)  );
  draw(  ellipsoid( (-0.052997495101288455,0.08452941813354285,0.9950105743071245), (0.5052868025262137,-0.8571689844894544,0.09973253843231203), (0.8613225369777713,0.5080512862884238,0.0027162096483570562), 0.1616048639643841, 0.24731119586936107, 0.9369158307459865,  (gDx*0, gDy*1, gDz*2)), rgb(0.12061992894016174,0.6257090294857688,0.5335353027024131)  );
  draw(  ellipsoid( (-0.11471149494454386,0.1534163980063871,0.9814808616322245), (0.49190804201582744,-0.8495968441799091,0.19029366925805588), (0.8630572119667057,0.5046272002005886,0.02199176413960133), 0.16339842709140875, 0.25677850033360883, 0.8992780904986832,  (gDx*0, gDy*1, gDz*3)), rgb(0.11962144664834584,0.6171712339785846,0.536829913231365)  );
  draw(  ellipsoid( (-0.4528575555442325,0.7140976377048565,-0.5338394873094776), (-0.31954488842872053,0.428982643075586,0.8449052942306381), (-0.8323527489385609,-0.5532074256228192,-0.03391821885349738), 0.2934680658502307, 0.3348985212209121, 0.7830303820973328,  (gDx*0, gDy*2, gDz*0)), rgb(0.1525650778658448,0.4990359806807483,0.5576483068240807)  );
  draw(  ellipsoid( (0.5300015826476153,-0.8406265930179423,-0.11155829642868301), (0.048683981042557056,-0.10117488639452857,0.9936767645230024), (0.8465980110896028,0.5320813598258269,0.012697800794937968), 0.28935845902523, 0.31985212230285437, 0.7962569491805438,  (gDx*0, gDy*2, gDz*1)), rgb(0.1484012264837087,0.509690861792587,0.5571604801139842)  );
  draw(  ellipsoid( (-0.46588787223994704,0.7628478717215207,0.44834318898549286), (0.22326261083249113,-0.3889525371634981,0.8937951277772425), (-0.856213931855815,-0.5165065812204375,-0.010892862432875088), 0.2657222821523543, 0.33390920253037976, 0.7772738933378174,  (gDx*0, gDy*2, gDz*2)), rgb(0.14717507005729877,0.5128468767968749,0.556980593435844)  );
  draw(  ellipsoid( (-0.42615014900177584,0.7138743143524826,0.5556793264226565), (0.27208825941888254,-0.48466364466502215,0.8313056782113425), (-0.8627653386594567,-0.5054548593571838,-0.012302664753727661), 0.24863267795177532, 0.35680901621307337, 0.7343164086003959,  (gDx*0, gDy*2, gDz*3)), rgb(0.14943507317977903,0.5070361486096406,0.5572996125068617)  );
  draw(  ellipsoid( (0.48933222943977706,-0.8579432955629441,0.15648409130033736), (-0.8088681288886181,-0.37941050527056386,0.4491993082792551), (0.32601582676263213,0.346382693114033,0.879620776591451), 0.357197292830258, 0.564801572086313, 0.6183917104597565,  (gDx*0, gDy*3, gDz*0)), rgb(0.208762960835406,0.367452349807176,0.5526400842384014)  );
  draw(  ellipsoid( (-0.4868278968469911,0.8734467423200363,-0.009454479470156383), (0.8451919771526134,0.4682914417239457,-0.25759201727725295), (0.22056545651663328,0.13339383021198875,0.9662075167648946), 0.34489689131864754, 0.5738891627169133, 0.6116371593795678,  (gDx*0, gDy*3, gDz*1)), rgb(0.2023091671289328,0.3813501311251095,0.5540775023735875)  );
  draw(  ellipsoid( (-0.45382752784689434,0.8821252825574943,0.12607759849091793), (0.8905962777115644,0.45372046095831103,0.031241213710161083), (0.029645321632301068,-0.12646236270384362,0.9915283282512299), 0.32794369645185767, 0.5654024792581691, 0.6196883069389442,  (gDx*0, gDy*3, gDz*2)), rgb(0.19330331352402916,0.4010946332649816,0.5556888680636765)  );
  draw(  ellipsoid( (-0.41366060545807504,0.8785949943490322,0.2386540160920883), (0.9103335237276897,0.3953138419779301,0.12255546466060284), (-0.013333381776099124,-0.26795111913457603,0.9633402403538126), 0.31402592222592934, 0.538571127467642, 0.6380936023198363,  (gDx*0, gDy*3, gDz*3)), rgb(0.1861717445557062,0.41716426436954,0.5566772726098654)  );
  draw(  ellipsoid( (0.0906415677385665,-0.03601154041103165,0.9952322719622387), (0.554852242975126,-0.8280455531427218,-0.08049565445435962), (-0.8269964296557554,-0.5595031106953643,0.05507426312575446), 0.12088541033050013, 0.2564848251947056, 0.9599876147326335,  (gDx*1, gDy*0, gDz*0)), rgb(0.13303849574235144,0.6561652935556093,0.5190214631049459)  );
  draw(  ellipsoid( (0.054306255961255934,-0.039617866026748966,0.9977380694626007), (0.5613179957427019,-0.8251744934245073,-0.06331795209104525), (0.8258165281821783,0.5634868943401385,-0.022573916098873388), 0.1144052890813164, 0.26054691445253025, 0.9707605587834415,  (gDx*1, gDy*0, gDz*1)), rgb(0.13740487271539975,0.6623347596253732,0.5155218076970334)  );
  draw(  ellipsoid( (0.024047595628481147,-0.07614767548791784,0.996806523184051), (0.5693251272326381,-0.8185672412067991,-0.07626644822351614), (0.8217606784496505,0.5693410253449828,0.02366821102436595), 0.11354123743015185, 0.2651522275630057, 0.9726878427058856,  (gDx*1, gDy*0, gDz*2)), rgb(0.1379338788992136,0.6629993800729583,0.5151267574503957)  );
  draw(  ellipsoid( (0.005332806146191544,-0.11633464443490608,0.993195756979867), (0.5719606640739572,-0.8143507604399293,-0.09845728882610415), (0.8202637136498756,0.5685939583526304,0.06219606574536673), 0.11315867461866895, 0.26400296676792706, 0.9633608404583487,  (gDx*1, gDy*0, gDz*3)), rgb(0.13740602599763405,0.662336208559201,0.5155209464510877)  );
  draw(  ellipsoid( (0.07171966254333767,-0.07383830431438437,0.9946879886781814), (0.5852482899278358,-0.8044241939472542,-0.10191248858139966), (0.8076761288667713,0.5894485736756742,-0.014479290400428913), 0.13406225543557496, 0.26831506607442923, 0.9593687014264862,  (gDx*1, gDy*1, gDz*0)), rgb(0.12678456760976028,0.6450553834382436,0.5248370686827009)  );
  draw(  ellipsoid( (0.01446038172745667,-0.015868888182417284,0.9997695113115572), (0.5803036030795268,-0.8141212297419264,-0.02131552336707951), (0.8142718376643117,0.5804780802677629,-0.002563730765371835), 0.12622766813245379, 0.262640013930172, 0.9651770544723642,  (gDx*1, gDy*1, gDz*1)), rgb(0.13054360403516235,0.6521700393673964,0.5211863970665783)  );
  draw(  ellipsoid( (-0.02773398889970305,-0.001186364654667739,0.9996146349461963), (0.5779177374608099,-0.8159559497585586,0.015065748630151648), (-0.8156236353784075,-0.5781128614660933,-0.023315334430279586), 0.12504630150881624, 0.2673872175964227, 0.9528816718899136,  (gDx*1, gDy*1, gDz*2)), rgb(0.12999435309948798,0.6512506305639589,0.5216770879361939)  );
  draw(  ellipsoid( (-0.05348858935889402,-0.011418207766309757,0.998503177430999), (0.5772241481726733,-0.8163003764041388,0.02158652933644347), (0.8148320401007361,0.5775147790436767,0.05025362087855834), 0.12884780609990554, 0.27067202329794127, 0.9283873697090154,  (gDx*1, gDy*1, gDz*3)), rgb(0.12671450206210583,0.6449104778592785,0.5249094816848197)  );
  draw(  ellipsoid( (0.04467646653428017,-0.09361817693924208,0.9946052736058583), (0.6054570940307082,-0.7893794804951197,-0.10149750273354066), (0.7946230053464985,0.6067253684492085,0.0214151034148985), 0.19935098181569275, 0.29113190298761155, 0.8886245307442486,  (gDx*1, gDy*2, gDz*0)), rgb(0.12221410389064694,0.5872338971185247,0.5460847019791797)  );
  draw(  ellipsoid( (-0.07184388997999912,0.06349858124306014,0.9953925786606308), (0.5897139204028855,-0.8021541414281166,0.09373486796694322), (0.804410310448344,0.5937311374434049,0.020183876598896158), 0.19275765085753307, 0.2775263164887467, 0.8950029437703713,  (gDx*1, gDy*2, gDz*1)), rgb(0.12083642757457094,0.5947073040186197,0.5440905375674238)  );
  draw(  ellipsoid( (-0.15003840039340433,0.16226773643372117,0.9752731207821042), (0.5650417054496796,-0.7953930693401673,0.21926635936263755), (0.8113053367936796,0.5839683612648072,0.027651465273829924), 0.1841766875440996, 0.28073667793725376, 0.8734566876943434,  (gDx*1, gDy*2, gDz*2)), rgb(0.12070473118376282,0.5955392728133817,0.5438578664138519)  );
  draw(  ellipsoid( (-0.19170622204095347,0.2039272233439298,0.9600325057048952), (0.5476941802247458,-0.7894720838709874,0.2770648186550289), (0.8144199220173385,0.5789192658459691,0.03965695719084345), 0.1823945848493949, 0.28860013482491037, 0.8309354940352829,  (gDx*1, gDy*2, gDz*3)), rgb(0.12217471164413149,0.5874211500684123,0.5460372279685404)  );
  draw(  ellipsoid( (0.5665342573546552,-0.7986220563258954,0.20308064012457652), (-0.1693107794208928,0.12837508201550235,0.9771661569504969), (0.8064569394676318,0.5879818445048378,0.062486441066757964), 0.33324213830139954, 0.3885223367441989, 0.7283149036188088,  (gDx*1, gDy*3, gDz*0)), rgb(0.17551970351137253,0.44205971730587584,0.5577069391986491)  );
  draw(  ellipsoid( (0.5643242168288324,-0.7958925560058041,-0.21930165889722086), (0.09877095323286803,-0.1986427784336779,0.9750822249295797), (0.8196233751534884,0.571923146821928,0.03348786394065147), 0.31833526500126513, 0.3844809949333491, 0.7292168022656473,  (gDx*1, gDy*3, gDz*1)), rgb(0.17110894265969814,0.45269356731634414,0.5579678488412546)  );
  draw(  ellipsoid( (-0.5104222600987507,0.7494622742855354,0.42163422040725795), (0.22707669967879351,-0.35544044514685574,0.9066963452097908), (-0.8294005599580162,-0.5585413049869792,-0.011239295563841949), 0.29461323426815433, 0.39780414294645977, 0.7114992220750207,  (gDx*1, gDy*3, gDz*2)), rgb(0.16822419957471513,0.4597472818319812,0.558078640540865)  );
  draw(  ellipsoid( (-0.4768764219206743,0.722522618979339,0.5005496411739517), (0.28035589535118083,-0.4146969931302154,0.8656945049095528), (-0.8330602920305186,-0.5531613407966925,0.004804257490233055), 0.28189882103416747, 0.4183483901476558, 0.6744358983816938,  (gDx*1, gDy*3, gDz*3)), rgb(0.17060357791650665,0.4539262609252269,0.5579893185852465)  );
  draw(  ellipsoid( (0.06690869040560163,-0.018619000532261187,0.9975853647520031), (0.6048802738821762,-0.7943872573865851,-0.05539620537584608), (0.7935005238917181,0.6071261962071798,-0.04188914492526749), 0.10695155259135508, 0.25964475890944516, 0.9336373297543635,  (gDx*2, gDy*0, gDz*0)), rgb(0.13913274367995998,0.6645055818368567,0.5142314714559965)  );
  draw(  ellipsoid( (0.049630904891925516,-0.061504070640117856,0.9968721194688435), (0.6172360595694671,-0.7827991177631627,-0.07902650186090267), (0.7852110671957991,0.619227575713412,-0.0008884983537982387), 0.10535633803401401, 0.26679081598821397, 0.9539747298244103,  (gDx*2, gDy*0, gDz*1)), rgb(0.14193676419655527,0.6678688128378917,0.5121920816673843)  );
  draw(  ellipsoid( (0.035378286209223855,-0.10702555854694452,0.993626643504799), (0.6195632044968759,-0.7777790231416291,-0.10583584834268399), (0.7841491009358237,0.6193587982571716,0.03879260912817782), 0.10115113140726223, 0.2629948883532556, 0.9499346531601915,  (gDx*2, gDy*0, gDz*2)), rgb(0.14503580845773878,0.6713372116425416,0.5100230077044411)  );
  draw(  ellipsoid( (0.028478759106601727,-0.1526367837676348,0.9878719413571886), (0.6161660764285812,-0.7755057575813937,-0.13758701325603495), (0.7871012174672046,0.61261148552666,0.07196416651683817), 0.09770915017159708, 0.25583485393897554, 0.9307229011026166,  (gDx*2, gDy*0, gDz*3)), rgb(0.1462912442373431,0.6726979933764863,0.5091593982442183)  );
  draw(  ellipsoid( (0.04210725510235956,-0.03172028546277504,0.9986094344426679), (0.6115920198052555,-0.7895363811803442,-0.050867515205425845), (-0.7900520111855124,-0.6128834524465532,0.013845336359976682), 0.11758528811873215, 0.28341809071102764, 0.9416701442743513,  (gDx*2, gDy*1, gDz*0)), rgb(0.13239153229865813,0.6551985849776154,0.5195584641151402)  );
  draw(  ellipsoid( (0.018446022766325978,-0.05368650526860746,0.9983874515418091), (0.6176720012751948,-0.7846069303341362,-0.05360283306262367), (0.7862194624179689,0.6176647343208833,0.01868777385117143), 0.11295735171362811, 0.2865934944025365, 0.9563026346655411,  (gDx*2, gDy*1, gDz*1)), rgb(0.1362553926303892,0.6607717112774791,0.516421674769192)  );
  draw(  ellipsoid( (0.003753814519559818,-0.08529466745161582,0.9963486982883408), (0.6217712255488038,-0.7801421721906556,-0.06912838960357659), (0.7831899207447013,0.6197604463613048,0.05010526090087864), 0.11120253073562024, 0.28532642242081846, 0.9518920705072837,  (gDx*2, gDy*1, gDz*2)), rgb(0.13704790592660054,0.6618543437214157,0.5157995203946067)  );
  draw(  ellipsoid( (-0.011584539456585502,-0.11283148716520813,0.9935466038136642), (0.6190938754149894,-0.7810781700780859,-0.08148414355638821), (0.7852315402919244,0.6141546610842875,0.07890171353789258), 0.11311613137442106, 0.2798432565471485, 0.927246951298036,  (gDx*2, gDy*1, gDz*3)), rgb(0.13393699371639922,0.6575078511719463,0.5182756801330879)  );
  draw(  ellipsoid( (0.024680248935836598,-0.06545209049655905,0.9975504544443332), (0.620741399909849,-0.781180393691073,-0.06661311395379513), (0.7836268242922942,0.6208648938072523,0.021349095701769665), 0.14207742905345572, 0.2969403239278669, 0.9366639654371449,  (gDx*2, gDy*2, gDz*0)), rgb(0.12255764742625347,0.633946363065709,0.5300432000513036)  );
  draw(  ellipsoid( (-0.020186475148288467,-0.031588214185311984,0.9992970984375318), (0.6181751727617526,-0.7859432383453209,-0.012356450956789722), (0.7857811158344701,0.6174912232767791,0.03539247341006366), 0.13253425627387536, 0.2941525969583549, 0.9406608082126128,  (gDx*2, gDy*2, gDz*1)), rgb(0.12551026984143582,0.6421832275820666,0.5262380620236722)  );
  draw(  ellipsoid( (-0.04544507100542931,-0.02989830165345087,0.9985193223366039), (0.6163042441369039,-0.7874954598366598,0.0044698317067969435), (0.7861957925225438,0.6155948280280522,0.0542142372902374), 0.1315552706203309, 0.29539183534571456, 0.9278180830414023,  (gDx*2, gDy*2, gDz*2)), rgb(0.12511745292388313,0.6412568301167385,0.5266844910819776)  );
  draw(  ellipsoid( (-0.06471261978251976,-0.035436926452333956,0.9972745364665112), (0.6138938011838919,-0.7893005980446679,0.011788417804009228), (0.7867316427530258,0.6129835154145193,0.0728322189774774), 0.136209185548213, 0.29486449649968544, 0.8962526156430526,  (gDx*2, gDy*2, gDz*3)), rgb(0.122309683084483,0.6331438988973093,0.5304019153883469)  );
  draw(  ellipsoid( (0.010367560488573008,-0.08551673917203506,0.9962827916866265), (0.6077440562702814,-0.7906597374792274,-0.07419125014059577), (0.7940652843171403,0.6062541272854656,0.043775077289225084), 0.2217859436867616, 0.32159798659012856, 0.8605261201132502,  (gDx*2, gDy*3, gDz*0)), rgb(0.1288359526942045,0.5629378349063495,0.5512854077747673)  );
  draw(  ellipsoid( (-0.07716859991737775,0.024729769313578443,0.9967113151241381), (0.5975352343255748,-0.799114338803236,0.06609022059690307), (0.7981206994723742,0.6006702190293474,0.046889626197740265), 0.2129590744866919, 0.3132628964907672, 0.8652912564871154,  (gDx*2, gDy*3, gDz*1)), rgb(0.12642932426634299,0.570715384398986,0.5498241206998008)  );
  draw(  ellipsoid( (-0.13301840003957668,0.09223768803807188,0.9868121980175877), (0.5825708391991711,-0.7982229239467525,0.1531384373717806), (0.8018212535008568,0.5952582402548394,0.052443348892178866), 0.20725467715648047, 0.31495343101908174, 0.8461572282873826,  (gDx*2, gDy*3, gDz*2)), rgb(0.1266089812199093,0.5701176324536809,0.549941023831601)  );
  draw(  ellipsoid( (-0.1769805515883745,0.1367790313583116,0.9746637271080507), (0.5679114860725167,-0.7946124531213203,0.2146336258171658), (0.8038373146018919,0.5915087031694455,0.06295256729014459), 0.21179267815176, 0.32161881812376825, 0.8082109465552286,  (gDx*2, gDy*3, gDz*3)), rgb(0.13055856265741858,0.5577224783960673,0.552164411445653)  );
  draw(  ellipsoid( (0.07655774940331607,-0.03339836907739456,0.996505624645074), (0.6150935317973659,-0.7850146950032879,-0.07356545228528114), (0.784728525126583,0.618576169579677,-0.039555837505895725), 0.09044677019065513, 0.24224836768772254, 0.8752771339552523,  (gDx*3, gDy*0, gDz*0)), rgb(0.14762168200715195,0.6740696339942274,0.5082677282925296)  );
  draw(  ellipsoid( (0.06968029309784693,-0.0878272477834723,0.9936956431929137), (0.6197236674836741,-0.776771329962738,-0.11211100262018768), (0.7817206871475145,0.6236286358841294,0.0003029701595629261), 0.08815146243215893, 0.23976190368993633, 0.8850760478385666,  (gDx*3, gDy*0, gDz*1)), rgb(0.15110172157239415,0.6775404216381719,0.5059736382700275)  );
  draw(  ellipsoid( (0.06532175744978974,-0.13711632256856027,0.9883987970899931), (0.622214326547614,-0.7687739516759997,-0.14776989904363705), (0.7801169142138697,0.6246484813955302,0.03509807470058603), 0.08500972010959254, 0.23733394739213978, 0.8871109698198445,  (gDx*3, gDy*0, gDz*2)), rgb(0.15479757168243685,0.6810163743575537,0.5036043277223811)  );
  draw(  ellipsoid( (0.058713910043516025,-0.18253947415591534,0.9814438430915362), (0.6199993397959361,-0.7638724119938601,-0.17916405009735017), (0.7824023871512695,0.6190139566839867,0.0683244173808338), 0.08345198377709932, 0.237252263089845, 0.8829449249546895,  (gDx*3, gDy*0, gDz*3)), rgb(0.1560875737310267,0.6821776044048312,0.5027938791267798)  );
  draw(  ellipsoid( (0.051533671898600954,-0.041769380982784876,0.997797373955736), (0.6159968575287659,-0.7850887883603608,-0.06467971788394035), (0.7860611631262118,0.6179732301671131,-0.014728700604315713), 0.10102343035960884, 0.28585186612928, 0.8940257643815411,  (gDx*3, gDy*1, gDz*0)), rgb(0.1393410004306445,0.6647672265946829,0.514075949869975)  );
  draw(  ellipsoid( (0.04194610070588384,-0.08916267566743462,0.9951334291960028), (0.625340408559856,-0.7744556050192183,-0.09574909543441591), (0.7792239075245091,0.626313446384588,0.023271631229791156), 0.10076293909153757, 0.2858224579967281, 0.9059882245786626,  (gDx*3, gDy*1, gDz*1)), rgb(0.1406519641046528,0.6663734102091012,0.5131109235048522)  );
  draw(  ellipsoid( (0.03823802993175961,-0.13559172111563886,0.9900266351123269), (0.6284440739730391,-0.7670290717501161,-0.12932304117335253), (-0.7769143446707023,-0.6271214302310631,-0.055882133010817535), 0.0975519435709408, 0.2791379370239671, 0.8975342357027458,  (gDx*3, gDy*1, gDz*2)), rgb(0.1426146560389428,0.6686578237116696,0.5117072783193852)  );
  draw(  ellipsoid( (0.03209789559527445,-0.17358142557595907,0.9842963038604651), (0.6248557262117471,-0.7651368683052646,-0.1553090280067514), (0.7800801539064719,0.6200282747222339,0.08390406442017864), 0.09608109345494834, 0.2714526787328527, 0.8780467850987201,  (gDx*3, gDy*1, gDz*3)), rgb(0.1420731825108381,0.6680275926411307,0.5120945202993941)  );
  draw(  ellipsoid( (0.04345361300701481,-0.07811590145811895,0.9959968320511979), (0.6205201235736295,-0.7792162713089635,-0.0881860463311719), (0.7829856702116613,0.6218680796332459,0.014612726523628656), 0.11685674404481979, 0.31355405548865617, 0.9105135090383771,  (gDx*3, gDy*2, gDz*0)), rgb(0.13008267135561588,0.6514098459885896,0.5215941371515747)  );
  draw(  ellipsoid( (0.018998086599389237,-0.09601952636142594,0.9951981326665004), (0.6265271264288705,-0.7745632353623361,-0.08669229650789473), (0.7791680387145923,0.6251656140434774,0.045443618510168614), 0.11194991689661624, 0.31188846900658884, 0.9194830815021878,  (gDx*3, gDy*2, gDz*1)), rgb(0.13361689090858972,0.6570295457388251,0.5185413760280189)  );
  draw(  ellipsoid( (0.006600631508503716,-0.1257984406035622,0.9920338623279958), (0.6300133425298476,-0.7698805140681974,-0.10181936108843431), (-0.7765562567503128,-0.6256666415910314,-0.07417299847075115), 0.11392953625751186, 0.30681760032160177, 0.9120121408588594,  (gDx*3, gDy*2, gDz*2)), rgb(0.13186968547205818,0.6543570796290646,0.5200133481989563)  );
  draw(  ellipsoid( (-0.005896789735514093,-0.1533698685508372,0.988151259317881), (0.6288179983157463,-0.7689128421289837,-0.11558964574436095), (0.7775301620290965,0.620685689080897,0.10097585109905773), 0.11845558692096729, 0.30092531780959725, 0.885200206188584,  (gDx*3, gDy*2, gDz*3)), rgb(0.12769593343634786,0.6469402183845422,0.5238951687369943)  );
  draw(  ellipsoid( (0.038868085657337105,-0.1225369678005948,0.9917025579474774), (0.6171761102324419,-0.7775790151761162,-0.12026854998749266), (0.7858644417936763,0.6167297355243847,0.04540388137963597), 0.14736174372364633, 0.32066098123964415, 0.903562566708005,  (gDx*3, gDy*3, gDz*0)), rgb(0.12034985607053551,0.6239310093548541,0.5342422438943613)  );
  draw(  ellipsoid( (0.0041906977726200514,-0.10622931459923406,0.9943328269608499), (0.6185579706812282,-0.7810134835583096,-0.0860463561508247), (-0.7857279904387392,-0.6154130898996957,-0.06243599779942597), 0.14195754292819435, 0.3184600109042923, 0.9073339524493906,  (gDx*3, gDy*3, gDz*1)), rgb(0.1212782718245153,0.6289896657210567,0.5321807064499452)  );
  draw(  ellipsoid( (-0.02263478745086974,-0.10341011188596012,0.9943812222466728), (0.6201101648226105,-0.7816338961049413,-0.06717020130597186), (0.7841881469907134,0.6151055203943006,0.08181777862865791), 0.14086576421560837, 0.3155065177397387, 0.8963251970202999,  (gDx*3, gDy*3, gDz*2)), rgb(0.12116506277331683,0.6284306388159473,0.5324118543105458)  );
  draw(  ellipsoid( (-0.04376154813084364,-0.11814016117559437,0.9920321714655208), (0.6167128294503356,-0.7843989761082445,-0.06620824927226875), (0.7859708728041724,0.6089015918830749,0.10718506661706487), 0.14829266021747048, 0.31381712671601125, 0.8712997202239474,  (gDx*3, gDy*3, gDz*3)), rgb(0.11970673359507793,0.6185643194437986,0.5363186367363765)  );
