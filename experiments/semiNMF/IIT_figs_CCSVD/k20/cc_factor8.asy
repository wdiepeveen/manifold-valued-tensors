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

  draw(  ellipsoid( (-0.385278104356851,0.16691649779498977,0.9075790131261557), (0.828427236958031,0.4958382268995852,0.26048563455814183), (-0.40653301879339826,0.8522225856599498,-0.32931378519854904), 0.7216183414857846, 1.0710186180128127, 1.4336143323495603,  (gDx*0, gDy*0, gDz*0)), rgb(0.19315890740658775,0.4014157566793264,0.5557113201909436)  );
  draw(  ellipsoid( (0.40469084177711806,0.1687969641225271,-0.8987396216283924), (0.6710098110318691,0.6128959211188272,0.4172582214587025), (-0.6212657692813135,0.771923684556267,-0.13476894724072946), 0.7636144000592064, 1.277021588762199, 1.427880781864086,  (gDx*0, gDy*0, gDz*1)), rgb(0.19625499708964342,0.3945644957009811,0.555207720660743)  );
  draw(  ellipsoid( (-0.31265143336523776,-0.3578179947023419,0.8798950868608468), (-0.17048881037914795,0.9324301832099541,0.31860244659222947), (-0.9344424255940884,-0.050400755016636774,-0.35252959754265945), 0.8090297372401057, 1.436505160773, 1.4764734472768912,  (gDx*0, gDy*0, gDz*2)), rgb(0.19454678882477436,0.3983354951543339,0.5554916352441137)  );
  draw(  ellipsoid( (-0.13213270551541462,-0.46513997983411587,0.8753203683755423), (-0.09587574045022947,0.8849198296229907,0.4557682936900475), (-0.9865844061844363,-0.02370009071576301,-0.1615224912319263), 0.9246072334488569, 1.4048229097818485, 1.5395673628146762,  (gDx*0, gDy*0, gDz*3)), rgb(0.21657870744186153,0.35074707987814696,0.550513891310804)  );
  draw(  ellipsoid( (-0.028870614835476035,0.47985647512563673,0.8768718554492559), (-0.6565540893836912,-0.6705759799552757,0.34534704692609397), (0.7537262203960893,-0.5657434209843013,0.3344116718961909), 0.4438393021376282, 0.6728490024878345, 0.8823082958778182,  (gDx*0, gDy*1, gDz*0)), rgb(0.1927161965308268,0.4024002359417759,0.5557801524613574)  );
  draw(  ellipsoid( (0.27732026954582356,0.12554307235516057,-0.9525399755824732), (0.4039983873068641,0.8842803385610105,0.2341657231276121), (-0.8717102564226747,0.44976351543678045,-0.19450966307477743), 0.5708761908302005, 0.805141290181018, 1.0522893014271304,  (gDx*0, gDy*1, gDz*1)), rgb(0.20468290875162426,0.37622087079133243,0.5535798890050888)  );
  draw(  ellipsoid( (0.3282351683764514,0.3898639955169794,-0.8603881329030633), (0.34947797027920524,0.7960983219359712,0.4940572943498132), (0.8775686996072787,-0.46285367753780216,0.12505858890664878), 0.6792372751635566, 1.1266067823639792, 1.2639129362732424,  (gDx*0, gDy*1, gDz*2)), rgb(0.1972765884607992,0.392318260169245,0.5550311280081839)  );
  draw(  ellipsoid( (0.3065534111618709,0.4721412354666726,-0.8265032727563908), (0.6882154910959845,0.4899038517537184,0.5351202237370436), (-0.6575594603688172,0.7328552857207742,0.17475321533692156), 0.8777901303743867, 1.4064067740348327, 1.4589861314128307,  (gDx*0, gDy*1, gDz*3)), rgb(0.2124910400314225,0.35947789927078244,0.5516834539902636)  );
  draw(  ellipsoid( (0.11428985705751496,0.6285497680613658,0.7693263401468691), (-0.3655482347264078,-0.6934627707964087,0.620873476328128), (0.9237490550264073,-0.3521854264295595,0.15050949720349524), 0.2503026155429627, 0.48753196050031805, 0.6281077140614344,  (gDx*0, gDy*2, gDz*0)), rgb(0.16288843115972235,0.4729731804135055,0.5581440266009965)  );
  draw(  ellipsoid( (0.03715163932647632,0.4239916064077359,0.9049037923399055), (-0.343438480013043,-0.8449543850830391,0.41000255801064694), (0.9384400706180902,-0.32601105015739956,0.11422359228103156), 0.3818368265357104, 0.5103680082859648, 0.7576496668094224,  (gDx*0, gDy*2, gDz*1)), rgb(0.19332112801250356,0.40105501825790946,0.5556860982837702)  );
  draw(  ellipsoid( (-0.16228399149738582,-0.15876862524999172,0.9738872777379856), (-0.38788713562200483,-0.8972514930556732,-0.2109107115068405), (0.9073078177347714,-0.4119857786933822,0.08402524639520081), 0.5322937974503426, 0.6740531389056102, 0.9187598610934435,  (gDx*0, gDy*2, gDz*2)), rgb(0.2158909773989186,0.3522164269366195,0.5507199370234321)  );
  draw(  ellipsoid( (-0.27805560330584517,-0.32057798496089734,0.9054914892082825), (0.620165111110985,0.6599578433306907,0.4240882926785447), (-0.7335395806845892,0.6794743560905837,0.015306305379270751), 0.7402747588767465, 0.9410517059457714, 1.1382368264622582,  (gDx*0, gDy*2, gDz*3)), rgb(0.2382403570799438,0.30337795526167155,0.5414690800130711)  );
  draw(  ellipsoid( (0.17524578250611325,0.6771628662126347,0.7146603167495097), (-0.27117693829616446,-0.6646148527204382,0.6962400201652507), (0.9464417487886608,-0.31581252376383045,0.06716000285721838), 0.18016046455709106, 0.4637097315998621, 0.5745218219383155,  (gDx*0, gDy*3, gDz*0)), rgb(0.13938319844584926,0.5331862186603848,0.5553593877890219)  );
  draw(  ellipsoid( (0.17715737554394106,0.6220790247840504,0.7626486420456721), (-0.2846353163045234,-0.709405763625054,0.6447683299044532), (0.9421241961666237,-0.33130220261760823,0.05138919670625798), 0.2515991102965656, 0.485058685877276, 0.6457865319843105,  (gDx*0, gDy*3, gDz*1)), rgb(0.16136419528469353,0.4767875145944919,0.5581268347204587)  );
  draw(  ellipsoid( (0.20091630375774136,0.5381822841861468,0.8185306761951616), (-0.41940585996674923,-0.7078579482529729,0.5683624281395131), (0.8852860348553335,-0.45748984041536234,0.08349660116943455), 0.3660260646874027, 0.5643164932791233, 0.7339154336085715,  (gDx*0, gDy*3, gDz*2)), rgb(0.19125717523481095,0.40565990570287375,0.5559972141747351)  );
  draw(  ellipsoid( (0.27275861170547605,0.44807961571830257,0.8513679567133352), (0.6943844923049914,0.5208134919811823,-0.4965717303841471), (-0.6659075886238578,0.726620922178893,-0.16908317144250093), 0.5070176499279625, 0.6973596319914757, 0.868676565742278,  (gDx*0, gDy*3, gDz*3)), rgb(0.21697707635829114,0.34989577380308934,0.5503916833463367)  );
  draw(  ellipsoid( (-0.696010311210883,0.15545111026935843,0.7010025670453378), (0.7158600012673153,0.07435274644945261,0.6942738131897119), (-0.055804168963344385,-0.9850414313172047,0.16303151019009582), 0.7526047901078243, 1.1977952733861092, 1.7546824781130492,  (gDx*1, gDy*0, gDz*0)), rgb(0.17337590335659833,0.4472080952558367,0.5578457125323706)  );
  draw(  ellipsoid( (-0.6291921386830354,0.15515209024927368,0.7616069074731067), (0.7615958580121185,-0.07259990539690708,0.6439728276838594), (0.15520631971106458,0.9852193068691059,-0.07248389955090327), 0.7757660565312127, 1.094175071499496, 1.3280379239069933,  (gDx*1, gDy*0, gDz*1)), rgb(0.21641664383753487,0.3510934064242679,0.5505636076977563)  );
  draw(  ellipsoid( (-0.3738865360581134,0.2816653319906694,0.8836704696373253), (-0.6382126006958204,0.6131955156142562,-0.46548462481981495), (-0.6729736506487755,-0.7380080625475269,-0.04950318320376591), 0.8397952155176612, 0.954138381734239, 1.0662711899556936,  (gDx*1, gDy*0, gDz*2)), rgb(0.2742781796916135,0.199040599816735,0.4985004429126727)  );
  draw(  ellipsoid( (-0.16638657902709955,0.9292108432389133,0.32997380975902685), (0.470015791420527,-0.219439178945915,0.854945379868725), (0.8668336992382071,0.2973443383533172,-0.4002320355921466), 0.8039173532830162, 0.8396398257240507, 1.0694840832704888,  (gDx*1, gDy*0, gDz*3)), rgb(0.2628197938299862,0.24020485227633714,0.5199093133132975)  );
  draw(  ellipsoid( (-0.5562231946364258,0.12292689399076984,0.821890951697505), (0.8274633166531609,0.17349204990833703,0.5340458484174876), (0.07694294861648648,-0.9771333007079359,0.19821779765151085), 0.6378551747754346, 1.0233021109316849, 1.306752308875966,  (gDx*1, gDy*1, gDz*0)), rgb(0.1876518870885291,0.4137917902650132,0.5564920259772075)  );
  draw(  ellipsoid( (-0.5774200249753046,-0.0721022400355769,0.8132572666379136), (-0.806526221355266,0.20518599732349166,-0.5544494212899501), (0.12689195807424286,0.9760635089399746,0.1766308509062577), 0.7288907610437103, 1.1568324310506697, 1.28705466311561,  (gDx*1, gDy*1, gDz*1)), rgb(0.20609861306765248,0.37317281871894326,0.5532645838984656)  );
  draw(  ellipsoid( (0.5218172193547175,0.1999547017764608,-0.8292918104156082), (-0.7154107731559035,0.6320834744638605,-0.2977547765564249), (-0.4646441813230667,-0.7486578648135477,-0.4728817888388565), 0.8529373922147672, 1.1803920625884254, 1.3404051432339268,  (gDx*1, gDy*1, gDz*2)), rgb(0.23123874934437202,0.3190631015709028,0.5450301439647068)  );
  draw(  ellipsoid( (-0.24727662798738062,-0.4341963862960739,0.8662146196980406), (0.6616253272881921,-0.7287817771611453,-0.17643425848920383), (0.7078885474013443,0.5294814627588681,0.46748602658554816), 1.072211350398414, 1.1655110634897985, 1.308452066010842,  (gDx*1, gDy*1, gDz*3)), rgb(0.2791304551981985,0.1734992374673947,0.48201311273546144)  );
  draw(  ellipsoid( (0.32176562805255393,-0.2523500853186663,-0.9125712657341383), (0.8821510905994223,0.429992350018169,0.19213545294948067), (-0.3439132651675362,0.8668483220065166,-0.36096766153810844), 0.46521687131132583, 0.7469720622631155, 0.8436410777253128,  (gDx*1, gDy*2, gDz*0)), rgb(0.20215980939886574,0.38167390330969564,0.5541077178354268)  );
  draw(  ellipsoid( (-0.4577774760052569,-0.08326437930394423,0.8851592091829499), (-0.06938657688580868,0.9959140916077881,0.05779814084470444), (-0.8863550560640608,-0.03495948049048493,-0.46168446943070424), 0.5924908067718708, 0.9571164940723615, 0.9667243788982153,  (gDx*1, gDy*2, gDz*1)), rgb(0.2137287353434995,0.35683470760064,0.5513413479817757)  );
  draw(  ellipsoid( (0.5089702361261915,0.26192314243697445,-0.8199668079828532), (0.6390279537652325,-0.7531841272613586,0.15606711616533878), (0.5767083951567447,0.6034152284268989,0.5507245128539726), 0.7405265136387211, 1.1188078075495687, 1.2255762260670815,  (gDx*1, gDy*2, gDz*2)), rgb(0.21780135397249883,0.3481343097139778,0.5501388190157283)  );
  draw(  ellipsoid( (0.5739660584249457,0.2596678111401404,-0.776618047455646), (-0.661388708199779,0.7061987409698681,-0.25268224100320286), (-0.4828332428923684,-0.6586774370739137,-0.5770754659901768), 0.9841370583913149, 1.2751296957758715, 1.439710791481717,  (gDx*1, gDy*2, gDz*3)), rgb(0.24686442222118857,0.2831058834153226,0.5359003015531649)  );
  draw(  ellipsoid( (-0.08727530107315458,0.4594826869345068,0.8838883878805266), (-0.6961641211871816,-0.6627871692422304,0.2758055196321318), (0.7125577437613521,-0.5912603730128697,0.37772031069352535), 0.3158890176056406, 0.5688139738116473, 0.6254144418778007,  (gDx*1, gDy*3, gDz*0)), rgb(0.186229857883874,0.41703159250869604,0.5566701255378639)  );
  draw(  ellipsoid( (0.22971240848497734,-0.22742859228240506,-0.946313079694281), (0.4177326038171141,0.9012380228990056,-0.1151933061825444), (0.8790515804525951,-0.36884449500234434,0.30202989489495935), 0.4506680837025683, 0.6635029995115904, 0.75533182067037,  (gDx*1, gDy*3, gDz*1)), rgb(0.21772228792126355,0.3483032722182582,0.5501630741741541)  );
  draw(  ellipsoid( (0.4131063826277297,0.03559799588696734,-0.9099867577723663), (0.3128549476304232,0.932873200698082,0.17851995172103577), (0.8552572118230853,-0.3584415909296058,0.37423886425177055), 0.6219781331813432, 0.8503796160575963, 0.9216038940179043,  (gDx*1, gDy*3, gDz*2)), rgb(0.24141972835124265,0.2960571078640677,0.5395927980154411)  );
  draw(  ellipsoid( (0.5994940604796215,0.08646947528864135,-0.7956946030311901), (0.6680367844838899,0.4934988950087328,0.5569431705314148), (-0.440832991081748,0.8654373867984134,-0.23808486618315872), 0.8268720043626192, 1.091387279680204, 1.1334845354494427,  (gDx*1, gDy*3, gDz*3)), rgb(0.2557428645737284,0.2604361337533333,0.5282099337272207)  );
  draw(  ellipsoid( (-0.8558102769293988,0.3807507574701913,0.3501680033753815), (0.48541521293279777,0.3571567247060672,0.7980044768362148), (-0.17877595183757794,-0.8529173082328143,0.49048060548961403), 0.6196074119582555, 0.9287162358560651, 1.4205118563278272,  (gDx*2, gDy*0, gDz*0)), rgb(0.17538241001322724,0.4423886259159827,0.5577163140578076)  );
  draw(  ellipsoid( (-0.7432061610129753,0.581978779124511,0.3300686335917109), (0.6225570752950411,0.4208236777346995,0.6597955139722657), (-0.24508629140047736,-0.6958506541683968,0.6750737566081018), 0.5506966431786438, 0.68796477611718, 1.0840603879357193,  (gDx*2, gDy*0, gDz*1)), rgb(0.19296293702495482,0.40185154623710556,0.55574178948149)  );
  draw(  ellipsoid( (-0.37817589982131944,0.8327992856062509,0.40426270998949865), (-0.8444524058139642,-0.13140412246371252,-0.5192620638122426), (-0.3793192891342906,-0.5377530162567752,0.7529532325432948), 0.4600807193294984, 0.5691760173945751, 0.9120059570028582,  (gDx*2, gDy*0, gDz*2)), rgb(0.1915682104317687,0.40496395093424636,0.5559516221093641)  );
  draw(  ellipsoid( (-0.18777999149423522,0.9095856470635949,0.3706651122648669), (0.8520668162144199,-0.03686884593338351,0.5221329609455455), (0.48859064205185854,0.4138775650544999,-0.7681045147920372), 0.3806617116717054, 0.5230533137534881, 0.8815282648306786,  (gDx*2, gDy*0, gDz*3)), rgb(0.17336200468872925,0.4472415861294736,0.5578465437710277)  );
  draw(  ellipsoid( (-0.7977309822207851,0.28448096812941287,0.531691507151669), (0.5743455240804308,0.08981767658119877,0.8136706974829131), (-0.18371853192815488,-0.9544649620314551,0.23504071409110483), 0.6552674885974459, 1.0455310788662835, 1.3721356765519768,  (gDx*2, gDy*1, gDz*0)), rgb(0.18524531740475436,0.41928449501548354,0.556788790335158)  );
  draw(  ellipsoid( (-0.7991068203671716,0.39896947248574843,0.44972397052760343), (0.5593641727655537,0.21927939488123843,0.7993924375466819), (-0.2203179789962156,-0.8903594257405051,0.39839688895128333), 0.6809730278784536, 0.9751588034832976, 1.09194905041635,  (gDx*2, gDy*1, gDz*1)), rgb(0.22601001388508976,0.33048572435785817,0.547254709201578)  );
  draw(  ellipsoid( (-0.6783614818183344,0.671263323088587,0.2987160040260661), (0.692730567185938,0.44885136548227894,0.5644969556967887), (-0.24484701605590778,-0.5898626982449063,0.7694880999390157), 0.688348849546885, 0.8807072946416591, 0.9932300795659282,  (gDx*2, gDy*1, gDz*2)), rgb(0.24986116292758004,0.27569802297873247,0.5335591554934445)  );
  draw(  ellipsoid( (-0.3957582182373262,0.8763718156843174,0.2744956709527302), (-0.8217557285707329,-0.20449941246143097,-0.5318811078277863), (-0.4099914087621841,-0.4360647096212933,0.8010958830028381), 0.6559476060844486, 0.809813757082675, 1.0656638682130395,  (gDx*2, gDy*1, gDz*3)), rgb(0.22724779744767232,0.32779758274773174,0.5467555102609182)  );
  draw(  ellipsoid( (0.7145915228838357,-0.22237446644422437,-0.6632560230380175), (-0.6137977441910689,0.25551005501323565,-0.7469718475371668), (-0.3355760489792367,-0.9408848008448116,-0.0460923734548819), 0.6165879204034608, 0.9988885083281893, 1.219607643543289,  (gDx*2, gDy*2, gDz*0)), rgb(0.1914289515742686,0.40527554867912785,0.5559720349082908)  );
  draw(  ellipsoid( (-0.7747839005288868,0.17973321492588612,0.6061401479308635), (0.3243865038691886,-0.7099090451985682,0.6251420187867571), (0.542663158495921,0.6809736551421518,0.49172306984070685), 0.7223026003781092, 1.055726005067953, 1.2428588150756665,  (gDx*2, gDy*2, gDz*1)), rgb(0.21420918836911304,0.3558086639348001,0.5512085480388054)  );
  draw(  ellipsoid( (-0.8295555233711015,0.34459112930713764,0.43942529199777425), (-0.07788892555954934,-0.8506099676817702,0.5199961520584521), (0.5529655946979678,0.3971393162144108,0.7324680297440106), 0.8547111840870181, 1.0949644945217123, 1.2758923898017234,  (gDx*2, gDy*2, gDz*2)), rgb(0.24367244286738343,0.29076275893209647,0.538138022462919)  );
  draw(  ellipsoid( (0.7207256506266468,-0.6595550732333076,-0.21340487787537243), (0.6645446088004142,0.744984580951347,-0.05812432415994472), (0.19731953639490657,-0.09992536973637928,0.9752332649371365), 0.9503207268280698, 1.192700918696717, 1.3312520213536698,  (gDx*2, gDy*2, gDz*3)), rgb(0.2558642552544806,0.2601051143239478,0.5280833312819997)  );
  draw(  ellipsoid( (0.6064198044541631,-0.26952466013260795,-0.7480718403643976), (-0.6425435604037636,0.38808968883015443,-0.6606997551137498), (-0.46839384469635015,-0.8813301600445299,-0.062163938472838534), 0.5068199467001931, 0.8084989859243938, 0.9411102962943357,  (gDx*2, gDy*3, gDz*0)), rgb(0.1996455954465079,0.3871377908939185,0.5546005255857464)  );
  draw(  ellipsoid( (0.7054446696643838,-0.13708864167990345,-0.6953808469928313), (0.3503195928020924,-0.7854529011760732,0.510235164343919), (-0.6161363493100246,-0.6035482121226513,-0.5060687252759912), 0.6427379471845676, 0.9470411881366487, 1.1565320463971944,  (gDx*2, gDy*3, gDz*1)), rgb(0.20726854141465373,0.370658245898713,0.5529956591830332)  );
  draw(  ellipsoid( (0.7899742331540947,-0.13331735556200466,-0.5984707124484491), (0.18162036974328516,-0.8813907987752999,0.43607831994786356), (-0.5856233877219323,-0.453185108471354,-0.672062872961224), 0.809084831563539, 1.0995875323185536, 1.3680409368943485,  (gDx*2, gDy*3, gDz*2)), rgb(0.21948957257229462,0.3445216539383937,0.5495970384755142)  );
  draw(  ellipsoid( (0.8419239527684489,-0.3341615991917486,-0.4236745016877507), (-0.19054537953199385,-0.9187010197927977,0.34594926589123126), (-0.5048331566901673,-0.21053375467436608,-0.8371493427393675), 0.9976561631445346, 1.295506958862829, 1.500793993424522,  (gDx*2, gDy*3, gDz*3)), rgb(0.2417913651863019,0.29518685167194875,0.539356463930033)  );
  draw(  ellipsoid( (-0.7320229995231401,0.6221575869426927,0.2776009099026703), (0.6256860174524524,0.4527276189085905,0.6352595616296125), (-0.26955395699865864,-0.6387156175380427,0.7206823323624371), 0.3985441928374994, 0.5780947589910945, 1.2384477090864827,  (gDx*3, gDy*0, gDz*0)), rgb(0.14520348023035667,0.5179419181930243,0.556650511286291)  );
  draw(  ellipsoid( (0.523610957269379,-0.7685658143298632,-0.36761141776462686), (0.7897252278240524,0.27597002887953137,0.5478819286109727), (-0.31963358704255856,-0.5771889917554782,0.7514567438186992), 0.31816289613475973, 0.4330761968121568, 1.0521365024795017,  (gDx*3, gDy*0, gDz*1)), rgb(0.13883221621311795,0.5346532522409048,0.5552083883482514)  );
  draw(  ellipsoid( (-0.3460901748224524,0.8510304355304626,0.3949288400360185), (-0.8614181795122637,-0.12145912971626105,-0.4931596088634448), (-0.37172612350646284,-0.5108765776676397,0.7751288999214636), 0.24641399492019084, 0.3790329384883049, 0.885947013737231,  (gDx*3, gDy*0, gDz*2)), rgb(0.13474219009792218,0.5457541852301213,0.5539095599151822)  );
  draw(  ellipsoid( (-0.26919175943317364,0.8903653468764025,0.36713123775420675), (0.8690527261819607,0.060276396452727976,0.49103372098703735), (0.41507006127758106,0.45123863433036976,-0.7900003412142856), 0.20722457259536256, 0.3579566858543811, 0.8083390672996624,  (gDx*3, gDy*0, gDz*3)), rgb(0.13058645707126287,0.5576395606186755,0.5521778070681206)  );
  draw(  ellipsoid( (0.7733515113713872,-0.5525042996546369,-0.31091226853041715), (-0.5901027968963525,-0.4480572977702167,-0.6715827179208483), (-0.23174582834366886,-0.7028397091641807,0.6725401209350388), 0.45628428356566786, 0.7385879261073858, 1.2045831679898877,  (gDx*3, gDy*1, gDz*0)), rgb(0.16096847993538296,0.47777934819174833,0.5581201367626165)  );
  draw(  ellipsoid( (-0.6319467601254816,0.7038053519423396,0.3245016470593431), (-0.7170182833927304,-0.3720213927686713,-0.5894784683116085), (-0.29415654615773834,-0.6051926221458563,0.7397390191498439), 0.3918938111496917, 0.5750090035900124, 1.0170543228344382,  (gDx*3, gDy*1, gDz*1)), rgb(0.16200549007256565,0.475180152844524,0.5581376894054259)  );
  draw(  ellipsoid( (-0.4323415033172114,0.8292965414021093,0.3540452950229864), (-0.8281772390102687,-0.20989528038668226,-0.5196791626154778), (-0.3566556957273241,-0.5178911253245183,0.7775509610407297), 0.3363104884820385, 0.5065448992427891, 0.9315719730341792,  (gDx*3, gDy*1, gDz*2)), rgb(0.15601380938632084,0.49027354444939003,0.5579190967017779)  );
  draw(  ellipsoid( (-0.2952440902344242,0.8874519247045324,0.35392090715281044), (-0.8715617371333807,-0.09840794563963942,-0.4803082495648285), (-0.3914218511359459,-0.45027209282078196,0.8025235055000585), 0.30054276095965377, 0.49407895027494503, 0.9401291867635224,  (gDx*3, gDy*1, gDz*3)), rgb(0.14616897536858203,0.5154415507696117,0.5568210526513118)  );
  draw(  ellipsoid( (-0.7965414601044782,0.48009736334204584,0.3674618674728596), (0.5747243882535386,0.41264138921505306,0.7066958054589451), (-0.18765281740816414,-0.7741018057198201,0.6046096381137932), 0.5258459841752549, 0.9628143574922037, 1.1788799840782462,  (gDx*3, gDy*2, gDz*0)), rgb(0.1742314612353671,0.4451465033769,0.5577945441190423)  );
  draw(  ellipsoid( (-0.7359460815893126,0.6095424510028736,0.294688590581221), (0.6683601631171673,0.5846075981198051,0.4599224375681034), (-0.10806506073347985,-0.535436230232884,0.8376335630827285), 0.5224145798888352, 0.8588147231550081, 1.0838331391587206,  (gDx*3, gDy*2, gDz*1)), rgb(0.18550781865397073,0.4186808054718996,0.5567585504445668)  );
  draw(  ellipsoid( (-0.5908751398303004,0.7632118474197719,0.26148469379415307), (0.7893947482619539,0.4800528511693005,0.3826293134362182), (-0.16650075239159984,-0.43250079311178685,0.8861267197250665), 0.5112852698021192, 0.7861460108163277, 1.120012773430899,  (gDx*3, gDy*2, gDz*2)), rgb(0.18054656315388085,0.4301685985808551,0.5572895780454817)  );
  draw(  ellipsoid( (-0.40678079250864563,0.8635292873979694,0.2980713952260319), (-0.8807217075158396,-0.28405718434539473,-0.37899972286553213), (-0.2426080393482913,-0.4166877557928978,0.87607799505287), 0.49393705180232783, 0.7642269698453927, 1.2355983477859507,  (gDx*3, gDy*2, gDz*3)), rgb(0.16613251171002374,0.4649079159981745,0.558126105828253)  );
  draw(  ellipsoid( (-0.7861370915533298,0.4278305379219448,0.44603755907489556), (0.1802988614360169,-0.5315423258940193,0.8276201280467098), (0.5911690061640142,0.7310429444325357,0.34072777924093645), 0.5489372314010291, 1.1015082984862314, 1.1407665341695292,  (gDx*3, gDy*3, gDz*0)), rgb(0.17590180315528126,0.44114537150864513,0.5576803878769698)  );
  draw(  ellipsoid( (0.7855373393691567,-0.5021837180694113,-0.3615834644488163), (-0.4729645751273829,-0.8640313502626156,0.17249445335411726), (0.399043354952067,-0.035515335692390564,0.9162439968694233), 0.6262574527917647, 1.0580378010701557, 1.2738040930827161,  (gDx*3, gDy*3, gDz*1)), rgb(0.18669509323048786,0.41596946715618777,0.5566129085340416)  );
  draw(  ellipsoid( (-0.7176025331841925,0.6403694258041175,0.2738130801531123), (0.6648768192669391,0.7469401564101243,-0.004383827493973615), (0.2073292540118438,-0.17890612409116674,0.96177293536134), 0.6887002429106419, 1.0391966052304509, 1.449047694644702,  (gDx*3, gDy*3, gDz*2)), rgb(0.1855513497724577,0.4185806944014588,0.5567535357017254)  );
  draw(  ellipsoid( (0.5759298087598747,-0.7746180482808301,-0.2612886041514471), (0.8173040705568001,0.5525717838298491,0.16333548287614613), (0.01785809712984917,-0.30762201317964283,0.9513410457739142), 0.7248491322575024, 1.0498514179722411, 1.6498663860723277,  (gDx*3, gDy*3, gDz*3)), rgb(0.17605216778105412,0.44078770900119685,0.557668982216355)  );
