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

  draw(  ellipsoid( (-0.3814985452801174,0.01287334953650554,0.9242797935803124), (-0.5874437861236415,0.768643305703753,-0.25317438010495485), (-0.7137006782215323,-0.6395480790906334,-0.2856739337734647), 0.8955099733113161, 1.0122942322934867, 1.4479239712618373,  (gDx*0, gDy*0, gDz*0)), rgb(0.2248139368865627,0.33307255695675386,0.5477140297424303)  );
  draw(  ellipsoid( (0.18392182863235404,-0.3569546110641871,0.9158363208524468), (0.8974783107278572,-0.31901148857756295,-0.3045724083507882), (0.40088083357227544,0.877960748436707,0.2616858450104914), 1.1085627783388112, 1.2263638879538903, 1.7176377660381426,  (gDx*0, gDy*0, gDz*1)), rgb(0.2327537913596805,0.3157113149628236,0.5443190139704998)  );
  draw(  ellipsoid( (-0.36192079618785705,0.3995217175615285,-0.8422563353774396), (0.929275216968275,0.08299956404274483,-0.3599425558297876), (0.0738979594800356,0.9129586351885833,0.40130489907212236), 1.194234414718229, 1.3724530213694888, 1.9969054670825885,  (gDx*0, gDy*0, gDz*2)), rgb(0.21873060129887495,0.3461463329746449,0.5498432019658553)  );
  draw(  ellipsoid( (0.05927096302556008,-0.4917313715136703,0.8687273514810661), (-0.9893160417572954,-0.1450531166567144,-0.014606945945370027), (0.13319430342060462,-0.8585801369805025,-0.49507517198789275), 1.174059552719448, 1.392301066001973, 2.2263174649180977,  (gDx*0, gDy*0, gDz*3)), rgb(0.19698095650401531,0.39296828426341224,0.5550822310547676)  );
  draw(  ellipsoid( (-0.30225779351000315,0.4281603877257804,0.8516565673115974), (0.9342110578104937,-0.04450880618473602,0.35393313724008363), (-0.18944636636014595,-0.9026060317516824,0.38653903518068944), 0.8994698667394396, 1.06116523182187, 1.2652311461035153,  (gDx*0, gDy*1, gDz*0)), rgb(0.2563639169278546,0.25874259027365587,0.527562217229089)  );
  draw(  ellipsoid( (-0.5594143755600706,-0.06842964530687863,-0.8260586783394454), (-0.828576113099192,0.01882295974847976,0.5595599351166435), (0.022741618635705414,-0.9974783605820637,0.06722900380301909), 0.956353167977529, 1.1064920208872389, 1.3225118048481672,  (gDx*0, gDy*1, gDz*1)), rgb(0.25958243210166143,0.2497627477277911,0.524002462982584)  );
  draw(  ellipsoid( (0.4970038573414667,-0.154559416172827,0.8538726794200748), (0.7850185297018444,0.4994012002028827,-0.3665301478196231), (-0.36977435526609526,0.8524727726492123,0.3695363284973587), 0.9206761164543741, 1.0770709139515378, 1.4648371033425256,  (gDx*0, gDy*1, gDz*2)), rgb(0.23001544016944112,0.3217531173535029,0.5455814233448307)  );
  draw(  ellipsoid( (-0.025512249215018643,-0.5534601073831198,0.8324848555231837), (-0.9088958437336603,-0.3338989350819647,-0.24983964135982872), (0.4162420814666256,-0.7630159963504013,-0.4945190784284621), 0.8000726277507914, 1.0846897450468573, 1.604487323503334,  (gDx*0, gDy*1, gDz*3)), rgb(0.19194151343971266,0.4041286692328098,0.5558969027229158)  );
  draw(  ellipsoid( (0.6524640213051666,0.5173786052448469,0.5537238298440731), (0.7364782330380024,-0.26072180336008505,-0.6241986490804725), (-0.17857915097657348,0.8150727084668961,-0.551149677264647), 0.9019143891574549, 0.9979657712784453, 1.4493868665279244,  (gDx*0, gDy*2, gDz*0)), rgb(0.22474578816205926,0.333219860881906,0.5477400162787607)  );
  draw(  ellipsoid( (0.7286586591180505,0.4121360269011486,0.5469921880817132), (-0.5001617393251614,-0.22537796247375755,0.8360879191487022), (-0.4678619379769131,0.8828072662599202,-0.04191106812255385), 0.7905774942306348, 1.018114705612901, 1.2875256181883155,  (gDx*0, gDy*2, gDz*1)), rgb(0.2270239763299086,0.3282836635764595,0.5468457774677729)  );
  draw(  ellipsoid( (0.4938273749920953,0.0488542895694879,0.8681864903919416), (0.5930115114940321,0.7113064829728789,-0.37733332017768706), (-0.635981030333436,0.7011821059264236,0.3222914572007873), 0.7566223449601359, 0.7992895748103115, 1.2864613984349487,  (gDx*0, gDy*2, gDz*2)), rgb(0.20954537935867565,0.36577722504326093,0.5524448957876495)  );
  draw(  ellipsoid( (-0.11373585687208228,-0.6565452833615846,0.7456624207758025), (-0.7502687727286451,-0.43523786668649,-0.4976592891431303), (0.6512763802947458,-0.6160489349863768,-0.4430832722777114), 0.5605480815923057, 0.8869466130299496, 1.342337842890747,  (gDx*0, gDy*2, gDz*3)), rgb(0.1706042665279126,0.4539245812534838,0.5579892893305135)  );
  draw(  ellipsoid( (0.8123861146039129,0.48793052995356156,0.31930017027554897), (0.5058719351156462,-0.3173618985303594,-0.8021066080163891), (0.2900385940889746,-0.8131452658046053,0.5046507610600114), 0.7664608391691314, 1.06526747471546, 1.41457997745807,  (gDx*0, gDy*3, gDz*0)), rgb(0.2046440918243699,0.37630462253111496,0.5535881586982865)  );
  draw(  ellipsoid( (0.7166520291855227,0.5471015001563451,0.43253880472270867), (-0.22946718265046145,-0.4006918685431598,0.8870123102696212), (-0.6586005474874221,0.7349326329739808,0.1616148008048766), 0.6748175196431325, 0.8827772846797461, 1.2421696314375608,  (gDx*0, gDy*3, gDz*1)), rgb(0.20492639354216646,0.375695652754366,0.5535277481308466)  );
  draw(  ellipsoid( (-0.21922779079940893,-0.6558251485162635,0.7223797826038818), (-0.6302846125605859,-0.4699739888595148,-0.6179528760066004), (0.7447687445848964,-0.5907773052251558,-0.3103251403292045), 0.5904933767358628, 0.6882032666143291, 1.2856892956496866,  (gDx*0, gDy*3, gDz*2)), rgb(0.17598757962181513,0.4409413406340769,0.5576738814446157)  );
  draw(  ellipsoid( (-0.1387108623394913,-0.7064547631899101,0.6940323942261793), (-0.6216793612072986,-0.4834046891975704,-0.6163072921122102), (0.7708917359252981,-0.5169541314547047,-0.37213486460420797), 0.4185390857199675, 0.8092253426039888, 1.350958251288649,  (gDx*0, gDy*3, gDz*3)), rgb(0.1437494153017786,0.5217167794417055,0.5563745035583281)  );
  draw(  ellipsoid( (-0.22138289240129272,-0.5059581011977439,0.8336642098497519), (0.683773009153393,-0.6900576479601183,-0.23722334295988573), (0.6953014360012151,0.5175198955828871,0.49872745139056235), 0.6553848898057126, 0.7693750077067116, 1.649071764718635,  (gDx*1, gDy*0, gDz*0)), rgb(0.1595534596493907,0.4813335857758006,0.5580832632938644)  );
  draw(  ellipsoid( (0.7792798798383754,-0.13939847087170615,-0.6109753965567821), (0.30680373962132235,-0.7652517626960651,0.5659162526778473), (0.5464378594437428,0.6284566858656181,0.553577329518961), 0.9063080502669016, 1.015065146379262, 1.9345410286507139,  (gDx*1, gDy*0, gDz*1)), rgb(0.17671783670282928,0.4392043258452929,0.5576184889985212)  );
  draw(  ellipsoid( (0.9197248841152207,-0.20956573728181085,-0.33194628977708873), (0.11440657013944483,-0.6657899074569585,0.7373160352504093), (0.3755226680883219,0.7161047415608874,0.5883678482597791), 1.0689506713796373, 1.3100263954955103, 2.1049387420457863,  (gDx*1, gDy*0, gDz*2)), rgb(0.19225631016372405,0.40342429788394996,0.5558507592865456)  );
  draw(  ellipsoid( (0.9799422193260207,-0.0404567148397359,-0.195132009180374), (0.12139156865101285,-0.6553604644379246,0.7454976517147148), (0.15804219007226983,0.7542320040120838,0.6373042839029693), 1.1886999278429977, 1.426936785223375, 2.1602894765252456,  (gDx*1, gDy*0, gDz*3)), rgb(0.20482973230318247,0.3759040830197639,0.5535486092049742)  );
  draw(  ellipsoid( (-0.6252208593834768,0.22972728088674033,0.7458714724456027), (0.3285923734837871,-0.7893766844641187,0.5185667769079761), (-0.70790248557275,-0.5693064433508725,-0.41804813655738504), 0.6477675326576854, 0.8319369280834816, 1.0898416553980124,  (gDx*1, gDy*1, gDz*0)), rgb(0.22084572384374024,0.3416144280247686,0.5491431519354492)  );
  draw(  ellipsoid( (-0.7820614655686119,0.18291739520237552,0.5957525414172004), (0.4383546858972735,-0.5180352559371588,0.7344961830792757), (-0.4429729488632203,-0.8355720795589695,-0.32495271415533955), 0.7768394522982796, 0.9434641783486901, 1.2058154998539696,  (gDx*1, gDy*1, gDz*1)), rgb(0.2363295129323108,0.30771227758107345,0.5425147604544365)  );
  draw(  ellipsoid( (-0.918851267846248,-0.005191751773924292,0.3945698839126734), (0.35696682787070144,-0.43711761179964986,0.8255318753715689), (-0.16818748876558104,-0.8993893701662747,-0.40349935496186234), 0.9031121992916235, 0.9838842687685416, 1.3415183094793122,  (gDx*1, gDy*1, gDz*2)), rgb(0.2412349896518256,0.29648964580725656,0.5397102066341501)  );
  draw(  ellipsoid( (-0.27453420853435123,-0.5617456315398465,0.7804311717187598), (0.9603753468624804,-0.11961181629316978,0.25173836922848947), (-0.048064139263981155,0.8186176511955677,0.5723241910559648), 0.9127548587824041, 1.0144414702877036, 1.5040431761977708,  (gDx*1, gDy*1, gDz*3)), rgb(0.21972382742283725,0.34402020024950514,0.5495210606438644)  );
  draw(  ellipsoid( (-0.6270509306977349,0.26645704711631546,0.731988915457849), (0.7055978462876471,0.592432357439184,0.3887873212609752), (0.3300587972245835,-0.7602792538936627,0.5594967796811319), 0.6334192145930637, 0.8184664285802443, 1.023386431799967,  (gDx*1, gDy*2, gDz*0)), rgb(0.22851985598674646,0.32502524192467847,0.5462258827852078)  );
  draw(  ellipsoid( (0.8719463347022626,0.07694566343971909,-0.48351727402142664), (0.44625125595857473,0.2813788519077063,0.849520899245272), (-0.201418484640027,0.9565068251596128,-0.21101015963311556), 0.7179445504798669, 0.7756283097487171, 1.0179138542291937,  (gDx*1, gDy*2, gDz*1)), rgb(0.25129025589447135,0.27208552949893594,0.5323575788676234)  );
  draw(  ellipsoid( (-0.23401981975909467,-0.3847760732865208,0.8928505459404317), (-0.9101906239911965,-0.2360962939486293,-0.3403109871606244), (0.3417422302824706,-0.8923037114177709,-0.29496836208589233), 0.7100984437854636, 0.7583899113924134, 1.0479468727175165,  (gDx*1, gDy*2, gDz*2)), rgb(0.241182915068878,0.2966099873870983,0.5397414623991662)  );
  draw(  ellipsoid( (-0.14611424645074897,-0.6139271923232775,0.7757216185656987), (0.9242642900599511,0.19483460967829053,0.3282910248415005), (0.3526842059181333,-0.7649397867790418,-0.5389628683855041), 0.5691191309258468, 0.8710854508113732, 1.1654861007233688,  (gDx*1, gDy*2, gDz*3)), rgb(0.18874374187418294,0.41131640317413376,0.5563494137386558)  );
  draw(  ellipsoid( (-0.9077850188284178,-0.4031609571169968,0.11570480649997252), (-0.13818960839859315,0.5479352895366892,0.8250276059690294), (0.39601766592076754,-0.7329584989184936,0.5531201019144177), 0.6623976286453183, 0.7168904814147578, 1.1742121281812175,  (gDx*1, gDy*3, gDz*0)), rgb(0.20350294615174847,0.3787667683573688,0.553831272341584)  );
  draw(  ellipsoid( (-0.8136520666391416,-0.5606644818051645,-0.15370638664674985), (-0.19275240160108637,0.010734439159575358,0.9811887094198256), (0.5484677074640468,-0.8279734963500197,0.11680352396677447), 0.6164145551116597, 0.7090525850796635, 1.0136076365608195,  (gDx*1, gDy*3, gDz*1)), rgb(0.22223772800834354,0.3386256437960911,0.548662195827566)  );
  draw(  ellipsoid( (-0.23971435180171052,-0.555662302541771,0.7961007694219765), (-0.7131674089971395,-0.45560422539838535,-0.5327448137180165), (0.6587330842492983,-0.6954597007254373,-0.28706537301160495), 0.5287505496802564, 0.6403525105866033, 1.017060488017687,  (gDx*1, gDy*3, gDz*2)), rgb(0.19552358705323647,0.3961765490698353,0.5553312405350084)  );
  draw(  ellipsoid( (-0.14098007496542408,-0.6611718514230848,0.7368693244724697), (-0.7287976492757419,-0.43444775729743845,-0.5292533727709052), (0.670058657785194,-0.6116428116678858,-0.42061201369208095), 0.3734895076641547, 0.7626555869053564, 1.095845080323143,  (gDx*1, gDy*3, gDz*3)), rgb(0.15004511680428914,0.5054730444861568,0.5573760271571134)  );
  draw(  ellipsoid( (0.0020584965248523265,-0.6671919239805425,0.7448830103896849), (0.5227217074010743,-0.6342982098181039,-0.5695856367862995), (0.8525008968932306,0.3905390090588786,0.3474500010066866), 0.45767036064527494, 0.7249282269123397, 1.6053151928261227,  (gDx*2, gDy*0, gDz*0)), rgb(0.1369019094448201,0.5398381390114664,0.5546363947835234)  );
  draw(  ellipsoid( (-0.4540681333836494,-0.224633962121932,0.8621842687656106), (-0.4737203243562946,0.8804461122508571,-0.02009223019274027), (-0.7545933902024685,-0.41755745291367213,-0.5061961961324791), 0.782132368832251, 0.7964642935348546, 1.6856076789476377,  (gDx*2, gDy*0, gDz*1)), rgb(0.17031797047846803,0.4546229190682074,0.558001452234575)  );
  draw(  ellipsoid( (-0.7495741691270495,0.47483250533243576,0.4611644574956052), (-0.22775162270811736,-0.8391811162406114,0.49386663432447114), (-0.6215044354880439,-0.26515871855340517,-0.7371723615571896), 0.8398691453598309, 1.2147337323716167, 1.700833921267897,  (gDx*2, gDy*0, gDz*2)), rgb(0.19080128805147986,0.4066799730852751,0.5560640388893254)  );
  draw(  ellipsoid( (0.8232192886860283,-0.42155155669251637,-0.38026870471468677), (-0.5008211451867909,-0.854683788384853,-0.1367252807856215), (0.26737274219534957,-0.3030014965676429,0.9147141137036793), 0.8862077636902852, 1.3278039538420754, 1.8351259240756974,  (gDx*2, gDy*0, gDz*3)), rgb(0.18762250762728644,0.41385839760149484,0.5564958633654667)  );
  draw(  ellipsoid( (-0.21232885795749878,-0.20932403740656827,0.9545155333687754), (0.366905165008337,-0.9223994569469576,-0.12066416913965333), (0.9057025206814153,0.32459619404387835,0.27265409375903277), 0.5062210777313872, 0.5830585191162364, 1.0841082906097883,  (gDx*2, gDy*1, gDz*0)), rgb(0.1776184494870038,0.43706519547901546,0.5575484676185805)  );
  draw(  ellipsoid( (0.5763954298413909,-0.44523158953451886,-0.6852278016386402), (0.05878181424524764,-0.8137740322599275,0.5782011092461307), (0.8150539900963587,0.37355141025029465,0.4428841125260884), 0.6044051082216316, 0.7928979550536672, 1.072550859010191,  (gDx*2, gDy*1, gDz*1)), rgb(0.21125463026032454,0.3621204499086932,0.5520089552805915)  );
  draw(  ellipsoid( (0.7229417554455765,-0.4365678274502325,-0.5355032682147258), (-0.5086724606877682,-0.8608282413774867,0.015068728705315417), (0.4675548587824962,-0.26150195196657194,0.8443987110047912), 0.6645566003261649, 0.9975784783223701, 1.1086254170704466,  (gDx*2, gDy*1, gDz*2)), rgb(0.21716905989283752,0.34948550898942066,0.5503327883974576)  );
  draw(  ellipsoid( (-0.8529195417871307,0.3270900258093189,0.40686652633718845), (0.4795795617080993,0.7988264237650825,0.3631525694340554), (0.20623214887475258,-0.5048647934949988,0.8382003585418906), 0.7353655712060385, 0.969885989631387, 1.2613374530767711,  (gDx*2, gDy*1, gDz*3)), rgb(0.21730948156004845,0.3491854308120423,0.5502897111255893)  );
  draw(  ellipsoid( (-0.24729806311015534,0.4868145987189252,0.8377679956026594), (-0.12774502238010121,-0.8734571773609237,0.4698444088991541), (0.9604815861009895,0.009170920936302304,0.2781924099793293), 0.4360188073827618, 0.6539389215615515, 0.8003170636010215,  (gDx*2, gDy*2, gDz*0)), rgb(0.2038076718754842,0.3781092894860477,0.5537663525134838)  );
  draw(  ellipsoid( (0.41099596028979446,-0.479591345458602,-0.7752899212466762), (-0.8070922320156484,-0.5868555236405082,-0.06482841500934375), (0.42389202592814845,-0.6523746896790455,0.628269698951572), 0.4931385328167304, 0.6995729597579358, 0.8078667511964048,  (gDx*2, gDy*2, gDz*1)), rgb(0.22322461261346319,0.3365014970246379,0.5483058774706556)  );
  draw(  ellipsoid( (-0.557702505696997,0.3926714825922531,0.7312845013386328), (0.780788911463522,0.5471475050978056,0.30165921733117207), (-0.28166751831707637,0.7392149311425247,-0.6117390740348797), 0.5556761798443801, 0.673807603802247, 0.8780109473021789,  (gDx*2, gDy*2, gDz*2)), rgb(0.23259974680553427,0.3160529440984574,0.5443924826018188)  );
  draw(  ellipsoid( (0.7317012035002594,-0.17054250335395738,-0.659945909409277), (0.5829482813970824,0.6583300189308962,0.4762067695766553), (0.353248708437073,-0.7331554001193819,0.5811183263874309), 0.6237539408979138, 0.6998214721467853, 0.8442378709034778,  (gDx*2, gDy*2, gDz*3)), rgb(0.26319442444375635,0.23906130768151748,0.5193995700191512)  );
  draw(  ellipsoid( (-0.035491924170013324,0.623036453774023,0.7813871643349409), (-0.8382123958941166,-0.44431620602272104,0.3162010253541563), (0.5441877458085165,-0.6437458243237505,0.5380065157389643), 0.4132837068488001, 0.6287962500101816, 0.8781948086911937,  (gDx*2, gDy*3, gDz*0)), rgb(0.18426206391139763,0.42154574083585095,0.5569020601887353)  );
  draw(  ellipsoid( (0.050033130202757535,0.6041440776456452,0.795302847554192), (-0.9009209267829985,-0.31640644878924423,0.2970327302656099), (0.4310895145634796,-0.7313664557595403,0.5284552372937106), 0.4334164914548231, 0.5529586384390939, 0.8409514260594055,  (gDx*2, gDy*3, gDz*1)), rgb(0.19582665955465636,0.3955066899615834,0.5552814746526729)  );
  draw(  ellipsoid( (0.07873075128762165,0.47521692772993335,0.8763391697285989), (-0.8471808317000552,-0.4314404635600189,0.3100705803573095), (0.5254389662003468,-0.76682983640573,0.3686270402402302), 0.4754030858640777, 0.5357654764482247, 0.7551828908451643,  (gDx*2, gDy*3, gDz*2)), rgb(0.22844779929029133,0.3251827101725332,0.546256610111626)  );
  draw(  ellipsoid( (-0.3479546291804258,-0.37891337828487737,0.8575268087871385), (0.4665390946355115,0.7234005855715397,0.5089527148679243), (-0.8131843881977745,0.5771622341694712,-0.0749326780511377), 0.4819850884102348, 0.5796536623772002, 0.7020731041248551,  (gDx*2, gDy*3, gDz*3)), rgb(0.24939346083874164,0.2768659761348796,0.5339375358581896)  );
  draw(  ellipsoid( (0.06627237163915138,-0.4962630169897783,0.865639064925749), (0.35112441907381087,-0.8004576925658805,-0.48577682606541167), (0.9339805217922915,0.33614059615034475,0.12120183386520905), 0.37252395350017115, 0.710195172436782, 1.6195899434512369,  (gDx*3, gDy*0, gDz*0)), rgb(0.12528674225211373,0.5747135330762894,0.5490002152747135)  );
  draw(  ellipsoid( (-0.07040326016981614,-0.2750023189633166,0.9588624017669373), (0.3640864319547156,-0.9020159540303853,-0.2319661370569807), (0.9287004097253029,0.33277761829704355,0.1636294769795527), 0.6058442676842901, 0.7256069455399005, 1.527276820662613,  (gDx*3, gDy*0, gDz*1)), rgb(0.16008761061872423,0.479991127493254,0.5580985143752457)  );
  draw(  ellipsoid( (-0.3855535460846992,0.616984762211791,0.6860599582397835), (0.06530933158952697,-0.723433505562705,0.6872980825206663), (0.9203712046282428,0.3097963302360361,0.23862749100521066), 0.6664293123148859, 1.0517866146771937, 1.3323369500840183,  (gDx*3, gDy*0, gDz*2)), rgb(0.19102114063007739,0.40618804338739345,0.556031812526414)  );
  draw(  ellipsoid( (-0.4023498711890355,0.6790125806194753,0.6140492622865426), (0.9102764601636643,0.36817632829852603,0.1893223635826598), (0.09752613611783727,-0.6351284174409902,0.7662248665586548), 0.5897715261398448, 1.0707327530605333, 1.5988861389056461,  (gDx*3, gDy*0, gDz*3)), rgb(0.1579454231738523,0.4853861408686797,0.5580218637477345)  );
  draw(  ellipsoid( (0.02502778148231419,-0.1965972760744559,0.9801648438880971), (0.30003913756491146,-0.933792346071302,-0.19495735520353585), (0.9535985140945731,0.29896716451603494,0.03561612637366052), 0.3723520801223695, 0.5650856602893046, 1.1928570084352121,  (gDx*3, gDy*1, gDz*0)), rgb(0.14337751329343495,0.5226833044351408,0.5563017515411592)  );
  draw(  ellipsoid( (-0.20882076345492992,0.39775199442392334,0.8934132524660082), (-0.28361271475877126,0.8496627408834261,-0.4445638928001507), (-0.9359261278419682,-0.3462175294323712,-0.06461969929120066), 0.49016471200871664, 0.7139133459916429, 1.1031785312232987,  (gDx*3, gDy*1, gDz*1)), rgb(0.1774155147423923,0.4375448046930327,0.5575655677818665)  );
  draw(  ellipsoid( (-0.3200170242254657,0.6126566889246907,0.7226623594194681), (0.9393409974314024,0.30453173667059463,0.15779389058189058), (0.12340014080403419,-0.7293231128081886,0.6729488854090766), 0.5009074084267151, 0.9557767609430184, 1.0778133977240203,  (gDx*3, gDy*1, gDz*2)), rgb(0.17600603481356766,0.4408974424756921,0.557672481556493)  );
  draw(  ellipsoid( (-0.33880394885739806,0.6655111983153583,0.6650614476538912), (0.934913816651747,0.317463387733134,0.15859745547348927), (0.10558427758097805,-0.6755085805271889,0.7297534638231226), 0.482559597146926, 0.7822115484030989, 1.4831232198988618,  (gDx*3, gDy*1, gDz*3)), rgb(0.1475234597138648,0.5119484367575176,0.5570357693730438)  );
  draw(  ellipsoid( (-0.026672046375817902,0.33387992370529335,0.9422381856455796), (0.282713516768759,-0.9015732716758675,0.32747320995145957), (0.9588334940845397,0.27511785174033976,-0.07034556329587177), 0.30884641597564766, 0.5504048464842441, 0.889888246533472,  (gDx*3, gDy*2, gDz*0)), rgb(0.15302889989512253,0.497853972258972,0.5576922512348768)  );
  draw(  ellipsoid( (-0.12754359566530105,0.5305342927852473,0.8380131236344831), (0.5462749156255432,-0.6676357092194894,0.5058124912784332), (0.8278383584752955,0.5222986922766002,-0.20466491707931117), 0.3444003803165721, 0.7156887788951953, 0.7768952472559496,  (gDx*3, gDy*2, gDz*1)), rgb(0.16830489361473777,0.4595494752575573,0.5580758799552853)  );
  draw(  ellipsoid( (-0.19368242012741613,0.6212617905913274,0.7592897389566414), (0.9704601243396923,0.23480903286162993,0.055424409362436394), (0.1438550214631944,-0.7475951482171643,0.6483881762971778), 0.3556240540589083, 0.6646696269642043, 0.9615141093152123,  (gDx*3, gDy*2, gDz*2)), rgb(0.15777513179426803,0.48581564711275066,0.5580148893567618)  );
  draw(  ellipsoid( (-0.20898612278680737,0.6611259400045635,0.7205812181396487), (-0.9681347445062337,-0.24384288132855073,-0.05705931742704191), (0.13798520559243715,-0.709544319037559,0.6910187713507785), 0.3652156656913649, 0.5760536940756557, 1.1634359179381812,  (gDx*3, gDy*2, gDz*3)), rgb(0.14431916537587805,0.5202360730344412,0.5564859589047445)  );
  draw(  ellipsoid( (0.0219531453208504,0.5118301846143843,0.8588061024050347), (0.25666866880301403,-0.8331069912180638,0.4899529932947381), (0.9662500990247868,0.20967260980202046,-0.14966009098420172), 0.2482874678263083, 0.6532513706282994, 0.7342195135485595,  (gDx*3, gDy*3, gDz*0)), rgb(0.14258917616626596,0.5247432213836231,0.5561322425813529)  );
  draw(  ellipsoid( (-0.007739251282230828,0.582895674037387,0.812510145892401), (-0.9852025827777491,-0.14357026490466157,0.09361329992718886), (-0.1712190844434189,0.7997625974148238,-0.5753814499083648), 0.25547991065492853, 0.5950059689986765, 0.754967286400354,  (gDx*3, gDy*3, gDz*1)), rgb(0.14653886902052923,0.514487535392048,0.5568798355982058)  );
  draw(  ellipsoid( (-0.04855709756131335,0.6175386445007863,0.7850402733774571), (-0.9788753033416706,-0.18570599341830066,0.08553610066091627), (0.19860853149895974,-0.7643031509529801,0.6135106720010444), 0.26640873618278793, 0.5092520683785576, 0.8601454774584373,  (gDx*3, gDy*3, gDz*2)), rgb(0.1437963050024859,0.521594919202579,0.556383676190458)  );
  draw(  ellipsoid( (-0.0954863219816858,0.6335764751535854,0.7677650763393501), (-0.9732303740001762,-0.22140404308366637,0.06166756708094692), (0.20905741182241566,-0.7413238832239006,0.6377569276173716), 0.2909387592580415, 0.4709820762498706, 0.8950568694483152,  (gDx*3, gDy*3, gDz*3)), rgb(0.14743268690066796,0.5121825250214028,0.5570213933003161)  );