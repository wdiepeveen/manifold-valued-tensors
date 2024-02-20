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

  draw(  ellipsoid( (0.22322179388817703,0.20167047252180595,0.9536776453530698), (0.5853729137709424,-0.8100391151276908,0.034280953698295556), (0.7794296520896936,0.5506048060778072,-0.2988708165200471), 0.10190674029489682, 0.23123330415176868, 0.8089407775719232,  (gDx*0, gDy*0, gDz*0)), rgb(0.132248381694422,0.6549816445028405,0.5196783543121128)  );
  draw(  ellipsoid( (0.18722790839777975,0.1440859102727846,0.9716918033913097), (0.5884490892560806,-0.8085081067152887,0.006504669810884835), (0.7865579315413975,0.5705733010197045,-0.2361625044176059), 0.10221216542623028, 0.22639063990209524, 0.8044868072811266,  (gDx*0, gDy*0, gDz*1)), rgb(0.13188751620548045,0.6543864869722372,0.5199975751694365)  );
  draw(  ellipsoid( (0.1486107944740222,0.0707526170675423,0.9863614443721398), (0.585179079341806,-0.8103475048530214,-0.030039415426379525), (-0.7971701680735902,-0.5816622633079919,0.16182933781285685), 0.10082915233328237, 0.21933930087416365, 0.7944112684267158,  (gDx*0, gDy*0, gDz*2)), rgb(0.13214213337916378,0.6548064144326963,0.5197723413497357)  );
  draw(  ellipsoid( (0.1076502927836481,-0.017534069353673534,0.9940341899932293), (0.5781174485636331,-0.8123182581858134,-0.07693674729439333), (-0.8088211360553519,-0.5829507730759663,0.07730954688952231), 0.10279902574999905, 0.21639891936347005, 0.7863883188731049,  (gDx*0, gDy*0, gDz*3)), rgb(0.13043729839027393,0.65199471474634,0.5212804348178721)  );
  draw(  ellipsoid( (0.17929179691801628,0.15363826068114206,0.9717251341880493), (0.5756327808982451,-0.8173840537578835,0.023026294050114672), (-0.7978103490888432,-0.5552284156247798,0.23498947501863862), 0.09792865988419493, 0.24675108010725405, 0.7929087926134637,  (gDx*0, gDy*1, gDz*0)), rgb(0.13292103520234247,0.6559897811480546,0.5191189592297389)  );
  draw(  ellipsoid( (0.14583882718506025,0.08491583681865161,0.9856573122250321), (0.5723193647719743,-0.8199105431052511,-0.014044429922785838), (0.8069582276625539,0.5661589900039037,-0.1681737697902668), 0.09571833584126696, 0.23808818312508365, 0.7762796586901367,  (gDx*0, gDy*1, gDz*1)), rgb(0.133116014950093,0.6562811246490251,0.5189571196041307)  );
  draw(  ellipsoid( (0.11704798719664944,0.008835774259514663,0.9930869537892683), (0.5654123701008092,-0.8226723578691156,-0.05932152505672078), (0.8164610342387434,0.5684471133635078,-0.10128799967664957), 0.08773960013625073, 0.22382909807005205, 0.7602845584721349,  (gDx*0, gDy*1, gDz*2)), rgb(0.13809321984154233,0.663199569128681,0.5150077651200743)  );
  draw(  ellipsoid( (0.08535426530492557,-0.07052975989384375,0.9938511972944305), (0.5605821919892597,-0.8212317541253307,-0.10642373814497011), (0.8236882027920354,0.5662190026708296,-0.0305579056166758), 0.08081179986553755, 0.21258779313204368, 0.751311599384499,  (gDx*0, gDy*1, gDz*3)), rgb(0.1439989313130828,0.6702133282056386,0.510736271517502)  );
  draw(  ellipsoid( (0.1626782740178518,0.10018500781220971,0.9815797182971132), (0.5638116054013727,-0.8258527374797942,-0.009150388617812286), (0.809723565655156,0.5549146062294609,-0.190833768017882), 0.11115394062975133, 0.27482620517725936, 0.8004949139915869,  (gDx*0, gDy*2, gDz*0)), rgb(0.12577869849478696,0.6428162747166838,0.5259329978943462)  );
  draw(  ellipsoid( (0.1253958913598676,0.030476432108576493,0.9916385720190576), (0.559101407720479,-0.8278631584626543,-0.045257118171806006), (-0.8195617647957004,-0.5601015782393135,0.12085005477362254), 0.10723031719726689, 0.27341560910689067, 0.7901440214582232,  (gDx*0, gDy*2, gDz*1)), rgb(0.1269079738751813,0.6453106052546339,0.5247095278518853)  );
  draw(  ellipsoid( (0.09409405257306176,-0.05064681159113845,0.9942742125520654), (0.5564734252826726,-0.8254498362274763,-0.0947095286980275), (0.8255202215721599,0.5621987801015618,-0.0494863155606464), 0.10361792824727391, 0.26517447389048304, 0.7804072733414791,  (gDx*0, gDy*2, gDz*2)), rgb(0.1281119328287272,0.6477947731476886,0.5234672886280337)  );
  draw(  ellipsoid( (0.061217708528888734,-0.12883969498242426,0.9897740778376183), (0.5542031816006765,-0.8203415860724583,-0.14106209860841865), (0.8301272346250059,0.5571714414405673,0.021183936765410204), 0.09643341934195235, 0.25320347892733447, 0.7787571597561944,  (gDx*0, gDy*2, gDz*3)), rgb(0.13254201097986554,0.6554234338981324,0.5194335618434449)  );
  draw(  ellipsoid( (0.1589992545033817,0.026991317956602874,0.9869096746016499), (0.5512506072719164,-0.8317200935997718,-0.06606401353785592), (0.819049452139315,0.5545386863487095,-0.1471218552524933), 0.11960811667007461, 0.2793170969173401, 0.800132971562995,  (gDx*0, gDy*3, gDz*0)), rgb(0.12268183048638756,0.6343474348570962,0.5298638367091487)  );
  draw(  ellipsoid( (0.10510525664730509,-0.009001919600151563,0.9944203590376739), (0.5443541514349631,-0.8363251338961711,-0.06510628409817618), (0.8322448214561777,0.5481598634130956,-0.08300193553352264), 0.11284606747573045, 0.2795002471380879, 0.805413927008034,  (gDx*0, gDy*3, gDz*1)), rgb(0.12532621278766323,0.6417491577126909,0.5264472394127268)  );
  draw(  ellipsoid( (0.07057368844299926,-0.08890504532915808,0.9935367368218304), (0.5462623542337465,-0.829947484680266,-0.11306906305397843), (0.8346357258388568,0.5507114177025207,-0.01000697584311215), 0.10899824992324303, 0.2809774849259101, 0.8144541905218192,  (gDx*0, gDy*3, gDz*2)), rgb(0.12765644285567287,0.6468585462125841,0.5239359823978849)  );
  draw(  ellipsoid( (0.04209235285929109,-0.1679695170092126,0.9848931288147261), (0.5484545233927433,-0.8200795064600196,-0.16330106813591147), (-0.8351202725361228,-0.5470428177391441,-0.05760456543720851), 0.10846213999257281, 0.27785611807344396, 0.8290972199065871,  (gDx*0, gDy*3, gDz*3)), rgb(0.12902324284822922,0.6494678094713704,0.5226006235943356)  );
  draw(  ellipsoid( (0.2642097842595617,0.22337077566029123,0.9382423388882191), (0.6063737698956893,-0.794964417805164,0.01850474550450083), (-0.750002694051165,-0.5640364092898761,0.3454835566439855), 0.13226247734384905, 0.25251663664063106, 0.8226147934211512,  (gDx*1, gDy*0, gDz*0)), rgb(0.12111426029107815,0.6281797758847849,0.5325155817506962)  );
  draw(  ellipsoid( (0.22486938728799727,0.15973115283568218,0.9612074268723191), (0.6156541880481528,-0.7879076114455689,-0.013096433289565302), (0.7552507394233322,0.5947163648664147,-0.2755154550303618), 0.15135196434445725, 0.2699606557649753, 0.8465596906425061,  (gDx*1, gDy*0, gDz*1)), rgb(0.1195060478736661,0.6152089205554423,0.537548484305181)  );
  draw(  ellipsoid( (0.16655186217137258,0.10044950600500456,0.9809028361415817), (0.6236111361720041,-0.7813063341254102,-0.025875917332051575), (0.7637863759255662,0.6160116143376386,-0.1927694554425731), 0.17429134331827775, 0.28755366161826335, 0.8530953616366386,  (gDx*1, gDy*0, gDz*2)), rgb(0.12042362754341887,0.5975224934146548,0.5432879056541983)  );
  draw(  ellipsoid( (0.11765164579524137,0.01572725585387011,0.9929303820837482), (0.616315984706937,-0.7851645709587678,-0.060590457217723354), (0.7786608358177446,0.6190874331905324,-0.10206886317172954), 0.18622424445679614, 0.29706600671895567, 0.8434040083365584,  (gDx*1, gDy*0, gDz*3)), rgb(0.12244616766792754,0.5861307707243288,0.5463643762604442)  );
  draw(  ellipsoid( (0.20735902716176088,0.16230786053808502,0.9647063761902259), (0.6151188611685172,-0.7884342701362071,0.00043394646318593876), (0.7606780005293139,0.5933191047676922,-0.2633275895693222), 0.09638807413315106, 0.23408079285370884, 0.7641844652124404,  (gDx*1, gDy*1, gDz*0)), rgb(0.13164900561899817,0.6539931233062077,0.5202085611357613)  );
  draw(  ellipsoid( (0.171762606751768,0.1026024501095398,0.979780763310526), (0.6161558946108983,-0.7872087183868953,-0.025580211725188047), (0.768667366588488,0.6080914165873266,-0.19843212596436502), 0.09099972381574341, 0.2277609277308474, 0.7575733245450158,  (gDx*1, gDy*1, gDz*1)), rgb(0.135019535225575,0.6590834376183148,0.5173918718554044)  );
  draw(  ellipsoid( (0.13870150499554182,0.03131361803118169,0.9898390524917516), (0.6141517099204781,-0.7868139803236515,-0.06116729165988497), (0.7769038355637876,0.6163953420436024,-0.1283635952800701), 0.09056144993998176, 0.22283630936389984, 0.7489220991429357,  (gDx*1, gDy*1, gDz*2)), rgb(0.13461794374245492,0.6585253433313414,0.5177104691378633)  );
  draw(  ellipsoid( (0.10771355081919401,-0.0551989332443561,0.992648411442141), (0.6090127609117023,-0.7855283800321163,-0.10976621160823907), (0.7858124763686475,0.6163588580792884,-0.05099519634798242), 0.09277523308086905, 0.2231304828070749, 0.7442665888762887,  (gDx*1, gDy*1, gDz*3)), rgb(0.132536521492045,0.6554152313713643,0.5194381182995073)  );
  draw(  ellipsoid( (0.17032927779691162,0.10797165540241349,0.9794539594870424), (0.6259629085565076,-0.779515792849837,-0.022925222118170944), (-0.761024555606887,-0.6170066858044188,0.20036061348475542), 0.10071538649949904, 0.26148565861225287, 0.7607203121611893,  (gDx*1, gDy*2, gDz*0)), rgb(0.12825208563722323,0.6480520738844983,0.5233340018914691)  );
  draw(  ellipsoid( (0.14204660772936076,0.03904606398947358,0.9890895642556901), (0.6238476758163227,-0.7793288111109683,-0.05882755775091159), (0.7685290096093662,0.6253974808546119,-0.1350598101936773), 0.0938188394351818, 0.2547653968621058, 0.751872443868336,  (gDx*1, gDy*2, gDz*1)), rgb(0.13192827457517364,0.6544537077273422,0.5199615204007892)  );
  draw(  ellipsoid( (0.11438663313424607,-0.03119061544116529,0.9929465462292584), (0.6196663324041136,-0.7789900433768133,-0.09585483193154733), (-0.7764852443189413,-0.6262600560695148,0.06977827403111998), 0.08777089023486821, 0.2438591425471537, 0.7402709641027991,  (gDx*1, gDy*2, gDz*2)), rgb(0.13557335698725395,0.6598399995715566,0.5169570998037349)  );
  draw(  ellipsoid( (0.08618182957547153,-0.10302879319514553,0.9909378184446167), (0.6128953720528941,-0.7786736913708099,-0.1342629706326396), (0.7854501608422834,0.6189122313702554,-0.00396165282712306), 0.08344472574291734, 0.23548447290197705, 0.7315306602303213,  (gDx*1, gDy*2, gDz*3)), rgb(0.13857921965581552,0.6638101582370346,0.514644831089492)  );
  draw(  ellipsoid( (0.1481864793015592,0.0675891547817229,0.9866470866059966), (0.6308013686826128,-0.7748250420016881,-0.04166278381511045), (-0.7616629179764175,-0.6285521938892386,0.15745392639332462), 0.1134214166690577, 0.28705715058460624, 0.7725138949272505,  (gDx*1, gDy*3, gDz*0)), rgb(0.12322265407562989,0.6360941230569813,0.5290827001646158)  );
  draw(  ellipsoid( (0.11445768114170167,-0.0030683431319799516,0.9934233863253314), (0.6275331745199463,-0.7749985436996998,-0.07469519489407177), (0.7701308681676012,0.6319555700630931,-0.08677904909393559), 0.10913149183522694, 0.2888771493962685, 0.7673781377679606,  (gDx*1, gDy*3, gDz*1)), rgb(0.12453862421119585,0.6398011913318018,0.5273742364985202)  );
  draw(  ellipsoid( (0.08400881928388981,-0.0811048198247102,0.9931588626618241), (0.6238774279399938,-0.7728821478962606,-0.11588848247449085), (0.7769938694671307,0.6293450513545524,-0.014329450304584742), 0.10430780576992814, 0.282249982681994, 0.7668711143253224,  (gDx*1, gDy*3, gDz*2)), rgb(0.12671310777270173,0.6449075942692674,0.5249109226880652)  );
  draw(  ellipsoid( (0.0626319574766125,-0.15465338910508208,0.9859815247462602), (0.6168889909292691,-0.770603451279271,-0.1600571577491754), (0.7845541477635029,0.6182658409735962,0.047139570556971516), 0.09922250824862297, 0.27294656798918865, 0.7667581461926016,  (gDx*1, gDy*3, gDz*3)), rgb(0.1295617704198107,0.6504564699373798,0.5220884784340891)  );
  draw(  ellipsoid( (0.2960963792803728,0.229999774641414,0.9270528775867924), (-0.6361085074981221,0.7715099902300584,0.011760172776231542), (-0.7125257194414564,-0.5931883669123276,0.37474612805781043), 0.23292888768399905, 0.3441090447894202, 0.8655003863843187,  (gDx*2, gDy*0, gDz*0)), rgb(0.1318766742137581,0.5538586579172398,0.5527686799414235)  );
  draw(  ellipsoid( (0.2354391780399194,0.19796994995874806,0.9515126338400441), (0.6605270042580869,-0.7507674247559624,-0.0072353694541228), (0.7129323040020873,0.6302032789811799,-0.30752456336306616), 0.29574081633222266, 0.39871424400901095, 0.911914820417941,  (gDx*2, gDy*0, gDz*1)), rgb(0.14437478719031926,0.520091519166402,0.5564968397257899)  );
  draw(  ellipsoid( (0.1569205016378969,0.17665350910399336,0.9716838446156013), (0.6750813415444211,-0.737318763964015,0.02502444015681486), (0.7208613864428409,0.6520387856775266,-0.2349559182227208), 0.3535259789459992, 0.44829399089363625, 0.9300623669628187,  (gDx*2, gDy*0, gDz*2)), rgb(0.157540003114838,0.48641025668267174,0.5580030255242238)  );
  draw(  ellipsoid( (0.0842091073507064,0.12873486708216553,0.9880972422978086), (0.6623212067269011,-0.7480961898161873,0.041020847167317534), (0.744472595438555,0.6509834289594786,-0.14826034487057288), 0.38126414560261246, 0.4721678303262311, 0.9165715566195263,  (gDx*2, gDy*0, gDz*3)), rgb(0.166384181507947,0.46428461379865116,0.5581224146712168)  );
  draw(  ellipsoid( (0.23124362578304922,0.18100264026256818,0.9559102623953195), (0.659171624510386,-0.751797947198794,-0.017106023150279176), (-0.7155551376204685,-0.6340645793652712,0.29316028758904855), 0.12513392334833237, 0.2560128130591137, 0.7636416058451634,  (gDx*2, gDy*1, gDz*0)), rgb(0.12039172499102253,0.6242066526787786,0.5341326480486339)  );
  draw(  ellipsoid( (0.1968936892488896,0.11927695535851628,0.9731422727712347), (0.6642702295584517,-0.7462585920077732,-0.04293222538999862), (0.7210949571702218,0.6548825251714828,-0.22616573783112717), 0.13362625958971697, 0.2630159426256027, 0.7780117342309427,  (gDx*2, gDy*1, gDz*1)), rgb(0.11970131531106577,0.6185122500181216,0.5363385085057509)  );
  draw(  ellipsoid( (0.15181038899926774,0.05420073851968952,0.9869224314685584), (0.66399729004349,-0.7452253922916157,-0.06121040351696159), (0.732162007076844,0.6646061951469666,-0.1491221001913517), 0.14968387323593862, 0.2754727715596128, 0.7812196721012421,  (gDx*2, gDy*1, gDz*2)), rgb(0.11971197012356141,0.6042087341390158,0.5412638614426973)  );
  draw(  ellipsoid( (0.11018622740267599,-0.024664848049981203,0.9936048714461081), (0.652648283356886,-0.7521705991088737,-0.09104728478932211), (0.7496060388778063,0.6585066705105346,-0.06678136993993872), 0.16161284113202662, 0.28690293369011505, 0.7708243343511058,  (gDx*2, gDy*1, gDz*3)), rgb(0.12143594900109107,0.5911858460907474,0.5450550057380256)  );
  draw(  ellipsoid( (0.1794760686754656,0.13019387567977156,0.9751091710717726), (0.684846883320668,-0.7281162576807626,-0.028835077716707444), (0.7062386899475042,0.67297568309445,-0.21984231345396554), 0.09723732817326688, 0.24989008978776797, 0.7383819309886182,  (gDx*2, gDy*2, gDz*0)), rgb(0.1286064207711491,0.6487025830823874,0.5229970255999628)  );
  draw(  ellipsoid( (0.15258631748918322,0.06427639712110664,0.986197728900356), (0.681719052194792,-0.7293159461319766,-0.05794294255777885), (0.7155253661399604,0.6811510812519859,-0.15510207903031475), 0.09520747381267097, 0.24287622453925195, 0.7342898405320997,  (gDx*2, gDy*2, gDz*1)), rgb(0.1295829829175015,0.6504954130833929,0.5220683051345175)  );
  draw(  ellipsoid( (0.12512821808066238,-0.005868405145483249,0.9921232236274922), (0.6758446505804966,-0.7315818133039516,-0.08956594623309048), (0.7263449162220965,0.6817284006581734,-0.08757539845392939), 0.09280668776579182, 0.23968406110899054, 0.729665207457738,  (gDx*2, gDy*2, gDz*2)), rgb(0.1307857831570671,0.6525694533667212,0.5209721659214859)  );
  draw(  ellipsoid( (0.09700430543376437,-0.08061708127720872,0.9920136344494752), (0.6675968529271424,-0.7339669585916854,-0.12492776175633097), (0.7381765416816553,0.6743837111778541,-0.017378245276682743), 0.09270693043946766, 0.24000385639603947, 0.7250824568475338,  (gDx*2, gDy*2, gDz*3)), rgb(0.13042582070902087,0.6519757851766222,0.5212905879507207)  );
  draw(  ellipsoid( (0.14327407740390483,0.08539398171706615,0.985992092580141), (0.7028732770084476,-0.7101537690906041,-0.0406298008094417), (0.6967364603718296,0.6988486904422975,-0.16176777384763583), 0.10290868399037904, 0.27470624334733984, 0.7438137783586709,  (gDx*2, gDy*3, gDz*0)), rgb(0.12584588425176102,0.6429747218511777,0.5258566425418214)  );
  draw(  ellipsoid( (0.12296983899797256,0.01615931638571461,0.9922788394351455), (0.6961650408892389,-0.7139903370151316,-0.07464606146812584), (0.7072712736570718,0.699969052989463,-0.09904882794126202), 0.09782949325003015, 0.26854187928884776, 0.737058923450683,  (gDx*2, gDy*3, gDz*1)), rgb(0.1280405080909657,0.6476528480791012,0.5235390495595925)  );
  draw(  ellipsoid( (0.09903768514364683,-0.05391021903059733,0.9936222749140948), (0.686534386614668,-0.7191147089846648,-0.10744566681594145), (0.720320812497466,0.6927970291549854,-0.03420823697615494), 0.09386330213826141, 0.2603040338584031, 0.7284148983807889,  (gDx*2, gDy*3, gDz*2)), rgb(0.129811839805534,0.6509155619662204,0.5218506599223128)  );
  draw(  ellipsoid( (0.08354687299832907,-0.12501831523381404,0.9886305381022273), (0.6752662007378109,-0.7224851192200965,-0.14842779607150908), (0.7328270451841439,0.6799894656254771,0.024059270251915583), 0.08897728865275113, 0.25406824965193286, 0.7237665161135871,  (gDx*2, gDy*3, gDz*3)), rgb(0.13291477022505727,0.6559804198659891,0.5191241593676506)  );
  draw(  ellipsoid( (0.3381924178019028,0.20914473433291322,0.9175425704794947), (-0.6597266683070053,0.747983834519964,0.07266984533731727), (-0.6711084947054701,-0.6299036937482,0.3909408202484511), 0.4638646038776777, 0.5358974148248675, 0.9215389161273475,  (gDx*3, gDy*0, gDz*0)), rgb(0.18836308058743043,0.41217941540580194,0.5563991336821382)  );
  draw(  ellipsoid( (0.24567792266986102,0.25049857327159614,0.9364255566255744), (0.7087092150084072,-0.7054951964231648,0.0027886172213447804), (0.6613422766425784,0.6629683194638601,-0.35085524153484754), 0.5622748918723918, 0.6194658860437469, 0.9724546654608444,  (gDx*3, gDy*0, gDz*1)), rgb(0.20938816397277588,0.36611381649445596,0.5524841160089192)  );
  draw(  ellipsoid( (0.1126088340974961,0.3213187110775401,0.9402518473231897), (0.7491241826633639,-0.6491223107037101,0.13211050183699669), (0.6527880279479317,0.689488587039803,-0.3138045234054255), 0.6304849859423376, 0.6850022692070945, 1.0,  (gDx*3, gDy*0, gDz*2)), rgb(0.22626032038277372,0.3299421242228674,0.5471537604232444)  );
  draw(  ellipsoid( (-0.012141384164801346,0.35148784894229823,0.9361137104200957), (0.7394252507155488,-0.6270623527984871,0.24503694477991228), (0.6731291743750308,0.6951612027067839,-0.2522855859070169), 0.6529672314159586, 0.7106674851495038, 0.9921148750379003,  (gDx*3, gDy*0, gDz*3)), rgb(0.235997858985882,0.3084613132564284,0.5426919312049309)  );
  draw(  ellipsoid( (0.25878808189291547,0.1797382911717426,0.9490642103445095), (0.6982134165472681,-0.7137574206445416,-0.05521204060929072), (0.6674779049703273,0.6769375829140472,-0.31020760018882576), 0.22976086192678977, 0.35413097328910276, 0.8058241559054887,  (gDx*3, gDy*1, gDz*0)), rgb(0.1365565707611843,0.5407714662467553,0.554529142020544)  );
  draw(  ellipsoid( (0.21599027146281624,0.12915407749971924,0.9678158021538064), (0.7100306437029722,-0.7011747031751159,-0.06488852464058167), (0.6702273402631094,0.7011941670411164,-0.2431502672656421), 0.2730100964542102, 0.3901077358396097, 0.8330777383406724,  (gDx*3, gDy*1, gDz*1)), rgb(0.14648935473541844,0.5146152246486757,0.5568719938123368)  );
  draw(  ellipsoid( (0.15453015458147232,0.09070950891299116,0.9838151332022651), (0.7143575246976405,-0.6981440206016958,-0.047835691757923016), (0.6825055005147477,0.7101878001563128,-0.1726833236773335), 0.31275318122719836, 0.4210163287753322, 0.8354080231388704,  (gDx*3, gDy*1, gDz*2)), rgb(0.15758234249326916,0.486303116871042,0.5580052600219203)  );
  draw(  ellipsoid( (0.10652839344110603,0.02917690279632735,0.9938814867649314), (0.699828266369896,-0.7122594128765006,-0.054101075391607564), (0.706322942414136,0.7013096585047407,-0.09629467226903575), 0.3309842306183422, 0.43499670817576996, 0.8204654082631964,  (gDx*3, gDy*1, gDz*3)), rgb(0.16473003046465282,0.46838523657918096,0.5581428155407154)  );
  draw(  ellipsoid( (0.19602041091139935,0.14578233585891454,0.9697027941888405), (0.7279561490548403,-0.6841914629942808,-0.04429319382244727), (0.6570052081676166,0.7145834818390813,-0.24023863953028518), 0.1332650522039865, 0.2838983693356184, 0.7434231329470726,  (gDx*3, gDy*2, gDz*0)), rgb(0.11942607604839246,0.6113294592315119,0.5389158649595619)  );
  draw(  ellipsoid( (0.16917477991752258,0.07909803841450667,0.9824069391849973), (0.7284996411695192,-0.6813997152356059,-0.07058824897046959), (0.6638284165774614,0.7276248541630588,-0.17289853946302716), 0.14406733873939412, 0.2901339813072767, 0.7584446987084759,  (gDx*3, gDy*2, gDz*1)), rgb(0.11974127038791725,0.6037509935086909,0.5414104393738601)  );
  draw(  ellipsoid( (0.13659335969172112,0.009315525135807556,0.9905834013749537), (0.7219997500245772,-0.6855995973482211,-0.09311043486315262), (0.6782762085261259,0.7279192352915267,-0.10037415922448542), 0.15509335626364085, 0.29741962516108517, 0.7567272909488205,  (gDx*3, gDy*2, gDz*2)), rgb(0.121092879692456,0.5930872128519462,0.5445436176384729)  );
  draw(  ellipsoid( (0.10917509771412583,-0.06729810141337499,0.9917417827162806), (0.7067443573463363,-0.6963290206692065,-0.12505322200158397), (0.6989944287318927,0.7145606066108169,-0.0284592354387179), 0.16113716968116523, 0.3014860442973117, 0.7417919832157274,  (gDx*3, gDy*2, gDz*3)), rgb(0.12284514455968121,0.5843429853467147,0.5468047950630068)  );
  draw(  ellipsoid( (0.14502513426380315,0.10909730698153634,0.9833948789988395), (0.7447746969910037,-0.6663490749625024,-0.03591045834538892), (0.6513665336457841,0.7376155420718804,-0.1778902778127064), 0.10624362637645748, 0.27128287243110133, 0.7169034826874291,  (gDx*3, gDy*3, gDz*0)), rgb(0.1228704332522083,0.6349565618110189,0.5295914290040985)  );
  draw(  ellipsoid( (0.1320938031325698,0.03784289950718021,0.9905145845119412), (0.7366968006882718,-0.6723191507915747,-0.07255882671709007), (0.6631960879179238,0.7392934968169175,-0.11668793654262619), 0.10442681776014855, 0.2655555549098399, 0.7187178211069885,  (gDx*3, gDy*3, gDz*1)), rgb(0.12363456569577969,0.6373299176055294,0.5285212269054292)  );
  draw(  ellipsoid( (0.11347959488465925,-0.03479422508542779,0.992930885533086), (0.7267398000057096,-0.6785539908400094,-0.10683512813095403), (0.6774744605019989,0.7337260002315587,-0.051715683807955645), 0.10327988040647879, 0.2637017884824264, 0.7175877705876219,  (gDx*3, gDy*3, gDz*2)), rgb(0.12404213770460759,0.6384440291146907,0.5280041351726722)  );
  draw(  ellipsoid( (0.09734949673509077,-0.10934386944623513,0.9892254514012211), (0.7124037060400449,-0.6864186747909214,-0.14598069228046737), (0.6949849171786634,0.71893902461391,0.011074465285792576), 0.10220196450784023, 0.2620568328023901, 0.7119063315071538,  (gDx*3, gDy*3, gDz*3)), rgb(0.12414835876741143,0.6387343878881635,0.5278693711745791)  );