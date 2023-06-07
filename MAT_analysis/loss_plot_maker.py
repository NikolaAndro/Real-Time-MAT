import matplotlib.pyplot as plt
# from learning.MAT_learning import abbrev_to_task
# import matplotlib as mpl
import argparse
import MAT_utils.utils_aux_MAT_attack as MAT_AUX
from matplotlib.ticker import MultipleLocator

parser = argparse.ArgumentParser(description='Run Adversarial attacks experiments')
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--testClean', type=bool, default=False)
args = parser.parse_args()

# copyt the data from the output txt files. Could've done this with json. Too late now.
results = {
    'test_results': {
           '2_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.057482483337835, 1.6719873012955655, 1.7096302828837915, 1.7168439901981156, 1.7222167037196996, 1.6989352582656232, 1.7018874809914029, 1.7085919884062304, 1.7106656413717367, 1.7028018211581044],
                                    'normal': [0.3422231049267287, 0.35082457114740745, 0.3642564108076784, 0.36655474040926117, 0.36805894442440307, 0.3633676059467276, 0.3640569494557135, 0.3647810722749258, 0.36547634816661323, 0.3721145925448113]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.0814998105629203, 1.4200549584074118, 1.5281260141392343, 1.6173551264497423, 1.6738585408200923, 1.6953587608239085, 1.702852010235344, 1.7062628645257851, 1.7062465896311494, 1.7115838281887095],
                                    'normal': [0.2922673722517859, 0.32525888030062017, 0.34725985871147863, 0.3688274564202299, 0.3623367377777689, 0.35893063062859565, 0.3663615887312545, 0.3745617415487152, 0.3768184171202256, 0.3781628898431345]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.1041493820160935, 1.6339177139026602, 1.65312754100131, 1.658142386515116, 1.6595340659937907, 1.6676589051472772, 1.670119223643824, 1.6716432060163047, 1.6745102690667222, 1.6767529200032814],
                                    'normal': [0.3300246169886638, 0.347245206599383, 0.3518711268287344, 0.352439781439673, 0.3526754198000603, 0.352919594651645, 0.35282778961142314, 0.35298536979046063, 0.3534351841690614, 0.3536302811706189]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.028387081991766, 1.547090113531683, 1.5949863578855377, 1.6158179941865587, 1.6619589891630349, 1.6655736714294276, 1.7004213153701468, 1.6975053145713412, 1.7024149761986487, 1.7125757532021435],
                                    'normal': [0.29395247510413536, 0.34281117913034775, 0.34176437375471763, 0.34511655390877083, 0.353580729617286, 0.3511239437098356, 0.3576238403308023, 0.3577928905941776, 0.35983746011847073, 0.3630908224693279]
                                },
           }, 
           '2_points_5_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.1218738142977056, 1.6677080380547906, 1.6997784592441676, 1.7105849054670825, 1.7138805996511401, 1.7069904000488754, 1.706759956694141, 1.703716875843166, 1.701049740044112, 1.7242404974612993],
                                    'normal': [0.3316511423931908, 0.37186298573140014, 0.3727930638593497, 0.37303979021986733, 0.37979782942644097, 0.3801705729101122, 0.3820594065582629, 0.38087092195589517, 0.38117558916204985, 0.37860409995329747]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.1337973383284106, 1.6092150909384502, 1.676007026003808, 1.6769315031385914, 1.676234197616577, 1.6797406727505713, 1.6836601579312196, 1.6928527974590812, 1.6994488755452264, 1.697665480240104],
                                    'normal': [0.30904505182172837, 0.3427108805511416, 0.3538515290034186, 0.37354462773529523, 0.37189136098340614, 0.3715857030804624, 0.37310867260411845, 0.37077624158760936, 0.373062776781849, 0.3704366102661054]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.0501654301722025, 1.7107831490408514, 1.710195165319541, 1.7412462158301443, 1.7510156255407432, 1.7246559612529795, 1.7581977165851397, 1.759352051105696, 1.7067319533259597, 1.776118222954347],
                                    'normal': [0.34013236158901883, 0.38763180706918854, 0.3890030801910715, 0.38685898750098713, 0.3881285353419707, 0.38424576090783186, 0.3918586657219326, 0.3905601308517849, 0.3849210178114704, 0.396578737263827]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.0565109294714388, 1.6303409817292518, 1.6649748524439703, 1.6941117586548795, 1.6914137761617445, 1.7045222545407481, 1.6970853286920136, 1.7028575614555594, 1.6676198357159329, 1.6900620249128833],
                                    'normal': [0.32295455938761997, 0.3665395888471112, 0.38640309062200723, 0.38812177531497993, 0.390170568965145, 0.38927188626269704, 0.3884876578739009, 0.39177207111083356, 0.3966857711678928, 0.39190258383750914]
                                },
           }, 
           '5_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.0681745144509778, 1.6200553132086686, 1.582614915887105, 1.58298257724526, 1.6324817524742834, 1.563879829092124, 1.642745912935316, 1.613683144825021, 1.5944064096077202, 1.5875890768680376],
                                    'normal': [0.3082017257656019, 0.3512688435537299, 0.3648729525276066, 0.3623403733231358, 0.36340341457386605, 0.3614476731143047, 0.35875659606505916, 0.3686393425022204, 0.36666723704829657, 0.36154256739567237]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.0530458214356728, 1.4962609104274474, 1.5170351488074076, 1.5241774588516077, 1.5275923913287133, 1.5368895252955328, 1.5704784823447158, 1.5836862999139374, 1.5790066802624574, 1.5946923167435165],
                                    'normal': [0.31931462472247096, 0.3341914008265918, 0.33340500838977777, 0.33491001242829355, 0.33354593361775897, 0.3338856005791536, 0.3416630691474246, 0.34458429487710146, 0.34741318619128353, 0.34602430404461537]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.0575275857423998, 1.6722344781934602, 1.6782611298806889, 1.6836647807937308, 1.684944219441758, 1.6526634339204769, 1.6701115787643748, 1.682510008517, 1.6803147923086108, 1.7036585788136904],
                                    'normal': [0.28634685987664255, 0.347442101725598, 0.34790225932278585, 0.34748688819482154, 0.35177282144113914, 0.3510158755115627, 0.35173952797639, 0.3485190726432604, 0.3502403815382535, 0.3528551603408204]
                                },
                                'POF-based-greedy':        {
                                    'zdepth': [1.1049068434951232, 1.563132699986094, 1.600042385661725, 1.6126589664478892, 1.6505921602249145, 1.6753942610062276, 1.6951732610918813, 1.6992259789987938, 1.7373498361135267, 1.727946020893215],
                                    'normal': [0.30820178318883956, 0.3434514228523392, 0.3534712562548746, 0.35365096536493795, 0.3572577449157066, 0.35925583145053114, 0.36437938858553304, 0.3688252801747666, 0.37157861111090357, 0.37365861479769047]
                                },
           }, 
           '3_points_3_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.0197926274279958, 1.584934410606463, 1.6194321379219134, 1.5813500286377582, 1.600242920757569, 1.6217247053519968, 1.5946104885376606, 1.5979767413483452, 1.5971017984999825, 1.5926305060534134],
                                    'normal': [0.3124103307109518, 0.35754515774471246, 0.3708703393174201, 0.3810851670417589, 0.368305677974347, 0.3718360502080819, 0.37477046171414485, 0.3740251626550537, 0.3749897020993773, 0.3758367668107613]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.1509047252615703, 1.5574920654296875, 1.6228267753247134, 1.627018292171439, 1.6256316720824882, 1.6197604584939702, 1.627216966864989, 1.6308016796701963, 1.6459725819912154, 1.6540683006502919],
                                    'normal': [0.31330102153660094, 0.3208603039844749, 0.347156644819938, 0.3482491232685207, 0.34852160633224805, 0.35474043927856325, 0.3536826352175978, 0.3579348233557239, 0.3711348551143076, 0.3722422357072535]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.9932249545436544, 1.4832410564127656, 1.489613993880675, 1.4889670389214742, 1.4953558169689376, 1.4957027415639346, 1.4996933678990787, 1.5114310173644232, 1.5361213551354163, 1.5571300538544803],
                                    'normal': [0.2923094980495492, 0.30608028867810044, 0.3061699933305229, 0.30667862609489677, 0.30865949917699875, 0.30906977960743853, 0.3094678116213415, 0.3103248262221051, 0.31615788180803517, 0.32675274002183347]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.0906920554097166, 1.5517956967206346, 1.5837280379128211, 1.6299365930950518, 1.6280626432182863, 1.6428285289056523, 1.6413225739272599, 1.6502504078383298, 1.6550571257306128, 1.6616724928629767],
                                    'normal': [0.3128962102624559, 0.3412199582328501, 0.35081258830950435, 0.37090757691983095, 0.37304771014095583, 0.3742851836779683, 0.37022044228524276, 0.3741223335880594, 0.3746446664185868, 0.3750189649075577]
                                },
           }, 
        },
    'clean_results': {
           '2_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817],
                                    'normal': [0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946],
                                    'normal': [0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873],
                                    'normal': [0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069],
                                    'normal': [0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574]
                                },
           }, 
           '2_points_5_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985],
                                    'normal': [0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066],
                                    'normal': [0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347],
                                    'normal': [0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184],
                                    'normal': [0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578]
                                },
           }, 
           '5_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212],
                                    'normal': [0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033]
                                },
                                'antipodes-greedy':           {
                                    'zdepth': [0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754],
                                    'normal': [0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396],
                                    'normal': [0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905],
                                    'normal': [0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055]
                                },
           }, 
           '3_points_3_predators' : {
                            'antipodes-non-greedy' :       {
                                    'zdepth': [0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761],
                                    'normal': [0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143],
                                    'normal': [0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678],
                                    'normal': [0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143],
                                    'normal': [0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157]
                                },
           }, 
        },
    
}

line_styles = [
    ['','-'],
]

# ===================================================== 2 x 2 TESTING + CLEAN
# Plot train fitness for every version for 2 x 2
epochs = range(1,11)
plt.figure(0)

plt.plot(epochs,results['test_results']['2_points_2_predators']['antipodes-non-greedy']['normal'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['2_points_2_predators']['antipodes-greedy']['normal'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['2_points_2_predators']['POF-based-non-greedy']['normal'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['2_points_2_predators']['POF-based-greedy']['normal'],'-', color='cyan' ,label='POF-based-greedy')

plt.plot(epochs,results['clean_results']['2_points_2_predators']['antipodes-non-greedy']['normal'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_2_predators']['antipodes-greedy']['normal'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['2_points_2_predators']['POF-based-non-greedy']['normal'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_2_predators']['POF-based-greedy']['normal'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/normal/2x2_loss_normal.pdf", bbox_inches='tight')
plt.close(0)

# ===================================================== 2 x 5 training

epochs = range(1,11)
plt.figure(1)
plt.plot(epochs,results['test_results']['2_points_5_predators']['antipodes-non-greedy']['normal'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['2_points_5_predators']['antipodes-greedy']['normal'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['2_points_5_predators']['POF-based-non-greedy']['normal'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['2_points_5_predators']['POF-based-greedy']['normal'],'-', color='cyan' ,label='POF-based-greedy')
 
plt.plot(epochs,results['clean_results']['2_points_5_predators']['antipodes-non-greedy']['normal'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_5_predators']['antipodes-greedy']['normal'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['2_points_5_predators']['POF-based-non-greedy']['normal'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_5_predators']['POF-based-greedy']['normal'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " --Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/normal/2x5_loss_normal.pdf", bbox_inches='tight')
plt.close(1)


# ===================================================== 5 x 2 training

epochs = range(1,11)
plt.figure(2)
plt.plot(epochs,results['test_results']['5_points_2_predators']['antipodes-non-greedy']['normal'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['5_points_2_predators']['antipodes-greedy']['normal'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['5_points_2_predators']['POF-based-non-greedy']['normal'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['5_points_2_predators']['POF-based-greedy']['normal'],'-', color='cyan' ,label='POF-based-greedy')
 
plt.plot(epochs,results['clean_results']['5_points_2_predators']['antipodes-non-greedy']['normal'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['5_points_2_predators']['antipodes-greedy']['normal'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['5_points_2_predators']['POF-based-non-greedy']['normal'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['5_points_2_predators']['POF-based-greedy']['normal'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/normal/5x2_loss_normal.pdf", bbox_inches='tight')
plt.close(2)

# # ===================================================== 3 x 3 training

epochs = range(1,11)
plt.figure(3)
plt.plot(epochs,results['test_results']['3_points_3_predators']['antipodes-non-greedy']['normal'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['3_points_3_predators']['antipodes-greedy']['normal'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['3_points_3_predators']['POF-based-non-greedy']['normal'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['3_points_3_predators']['POF-based-greedy']['normal'],'-', color='cyan' ,label='POF-based-greedy')
 
plt.plot(epochs,results['clean_results']['3_points_3_predators']['antipodes-non-greedy']['normal'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['3_points_3_predators']['antipodes-greedy']['normal'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['3_points_3_predators']['POF-based-non-greedy']['normal'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['3_points_3_predators']['POF-based-greedy']['normal'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/normal/3X3_loss_normal.pdf", bbox_inches='tight')
plt.close(3)











# ===================================================== 2 x 2 TESTING ADEPTH + CLEAN
# Plot train fitness for every version for 2 x 2
epochs = range(1,11)
plt.figure(8)

plt.plot(epochs,results['test_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['2_points_2_predators']['antipodes-greedy']['zdepth'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['2_points_2_predators']['POF-based-non-greedy']['zdepth'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['2_points_2_predators']['POF-based-greedy']['zdepth'],'-', color='cyan' ,label='POF-based-greedy')

plt.plot(epochs,results['clean_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_2_predators']['antipodes-greedy']['zdepth'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['2_points_2_predators']['POF-based-non-greedy']['zdepth'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_2_predators']['POF-based-greedy']['zdepth'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Z-Depth Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/zdepth/2x2_loss_zdepth.pdf", bbox_inches='tight')
plt.close(8)

# ===================================================== 2 x 5 training

epochs = range(1,11)
plt.figure(9)
plt.plot(epochs,results['test_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['2_points_5_predators']['antipodes-greedy']['zdepth'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['2_points_5_predators']['POF-based-non-greedy']['zdepth'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['2_points_5_predators']['POF-based-greedy']['zdepth'],'-', color='cyan' ,label='POF-based-greedy')
 
plt.plot(epochs,results['clean_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_5_predators']['antipodes-greedy']['zdepth'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['2_points_5_predators']['POF-based-non-greedy']['zdepth'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['2_points_5_predators']['POF-based-greedy']['zdepth'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Z-Depth Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " --Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/zdepth/2x5_loss_zdepth.pdf", bbox_inches='tight')
plt.close(9)


# ===================================================== 5 x 2 training

epochs = range(1,11)
plt.figure(10)
plt.plot(epochs,results['test_results']['5_points_2_predators']['antipodes-non-greedy']['zdepth'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['5_points_2_predators']['antipodes-greedy']['zdepth'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['5_points_2_predators']['POF-based-non-greedy']['zdepth'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['5_points_2_predators']['POF-based-greedy']['zdepth'],'-', color='cyan' ,label='POF-based-greedy')
 
plt.plot(epochs,results['clean_results']['5_points_2_predators']['antipodes-non-greedy']['zdepth'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['5_points_2_predators']['antipodes-greedy']['zdepth'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['5_points_2_predators']['POF-based-non-greedy']['zdepth'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['5_points_2_predators']['POF-based-greedy']['zdepth'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Z-Depth Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/zdepth/5x2_loss_zdepth.pdf", bbox_inches='tight')
plt.close(10)

# # ===================================================== 3 x 3 training

epochs = range(1,11)
plt.figure(11)
plt.plot(epochs,results['test_results']['3_points_3_predators']['antipodes-non-greedy']['zdepth'],'-', color='red' ,label='antipodes-non-greedy')
plt.plot(epochs,results['test_results']['3_points_3_predators']['antipodes-greedy']['zdepth'],'-', color='blue' ,label='antipodes-greedy')
plt.plot(epochs,results['test_results']['3_points_3_predators']['POF-based-non-greedy']['zdepth'],'-', color='green' ,label='POF-based-non-greedy')
plt.plot(epochs,results['test_results']['3_points_3_predators']['POF-based-greedy']['zdepth'],'-', color='cyan' ,label='POF-based-greedy')
 
plt.plot(epochs,results['clean_results']['3_points_3_predators']['antipodes-non-greedy']['zdepth'],'.', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(epochs,results['clean_results']['3_points_3_predators']['antipodes-greedy']['zdepth'],'.', color='blue' ,label='clean-antipodes-greedy')
plt.plot(epochs,results['clean_results']['3_points_3_predators']['POF-based-non-greedy']['zdepth'],'.', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(epochs,results['clean_results']['3_points_3_predators']['POF-based-greedy']['zdepth'],'.', color='cyan' ,label='clean-POF-based-greedy')

plt.xticks(epochs[::1])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Z-Depth Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("./plots_2/losses/zdepth/3X3_loss_zdepth.pdf", bbox_inches='tight')
plt.close(11)



















#======================================= loss space
# A = np.array([[1.8,1.0]])
# b = np.array([1.0]) 
# if args.num_POF_points == 5:
#     POF = [[0.7   , 0.79365079],
#         [1.05  , 0.52910053],
#         [1.45  , 0.38314176],
#         [1.9   , 0.29239766],
#         [2.5   , 0.22222222]]   
# elif args.num_POF_points == 2:
#     POF = [[0.7 , 0.79365079],
#             [2.5, 0.222222222]]
# elif args.num_POF_points == 3:
POF = [[0.7 , 0.79365079],
        [1.45  , 0.38314176],
        [2.5, 0.222222222]]

POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 4.5]
y_axis_height = 1.4

epochs = range(1,11)
plt.figure(12)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(results['test_results']['3_points_3_predators']['antipodes-non-greedy']['zdepth'], results['test_results']['3_points_3_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['3_points_3_predators']['antipodes-greedy']['zdepth'], results['test_results']['3_points_3_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(results['test_results']['3_points_3_predators']['POF-based-non-greedy']['zdepth'],results['test_results']['3_points_3_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['3_points_3_predators']['POF-based-greedy']['zdepth'],results['test_results']['3_points_3_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(results['clean_results']['3_points_3_predators']['antipodes-non-greedy']['zdepth'][:1],results['clean_results']['3_points_3_predators']['antipodes-non-greedy']['normal'][:1],'s', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(results['clean_results']['3_points_3_predators']['antipodes-greedy']['zdepth'][:1],results['clean_results']['3_points_3_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
plt.plot(results['clean_results']['3_points_3_predators']['POF-based-non-greedy']['zdepth'][:1],results['clean_results']['3_points_3_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(results['clean_results']['3_points_3_predators']['POF-based-greedy']['zdepth'][:1],results['clean_results']['3_points_3_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)

# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='   POF, POF points, and \nGAMAT Crossover Types')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined/3X3_combined_losses.pdf", bbox_inches='tight')
plt.close(12)



POF = [[0.7 , 0.79365079],
        [2.5, 0.222222222]]

POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 4.5]
y_axis_height = 1.4



epochs = range(1,11)
plt.figure(13)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(results['test_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'], results['test_results']['2_points_5_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['2_points_5_predators']['antipodes-greedy']['zdepth'], results['test_results']['2_points_5_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(results['test_results']['2_points_5_predators']['POF-based-non-greedy']['zdepth'],results['test_results']['2_points_5_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['2_points_5_predators']['POF-based-greedy']['zdepth'],results['test_results']['2_points_5_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(results['clean_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'][:1],results['clean_results']['2_points_5_predators']['antipodes-non-greedy']['normal'][:1],'s', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(results['clean_results']['2_points_5_predators']['antipodes-greedy']['zdepth'][:1],results['clean_results']['2_points_5_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
plt.plot(results['clean_results']['2_points_5_predators']['POF-based-non-greedy']['zdepth'][:1],results['clean_results']['2_points_5_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(results['clean_results']['2_points_5_predators']['POF-based-greedy']['zdepth'][:1],results['clean_results']['2_points_5_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='   POF, POF points, and \nGAMAT Crossover Types')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined/2x5_combined_losses.pdf", bbox_inches='tight')
plt.close(13)


POF = [[0.7   , 0.79365079],
    [1.05  , 0.52910053],
    [1.45  , 0.38314176],
    [1.9   , 0.29239766],
    [2.5   , 0.22222222]]   


POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 3]
y_axis_height = 1.4

epochs = range(1,11)
plt.figure(14)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(results['test_results']['5_points_2_predators']['antipodes-non-greedy']['zdepth'], results['test_results']['5_points_2_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['5_points_2_predators']['antipodes-greedy']['zdepth'], results['test_results']['5_points_2_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(results['test_results']['5_points_2_predators']['POF-based-non-greedy']['zdepth'],results['test_results']['5_points_2_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['5_points_2_predators']['POF-based-greedy']['zdepth'],results['test_results']['5_points_2_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(results['clean_results']['5_points_2_predators']['antipodes-non-greedy']['zdepth'][:1],results['clean_results']['5_points_2_predators']['antipodes-non-greedy']['normal'][:1],'s', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(results['clean_results']['5_points_2_predators']['antipodes-greedy']['zdepth'][:1],results['clean_results']['5_points_2_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
plt.plot(results['clean_results']['5_points_2_predators']['POF-based-non-greedy']['zdepth'][:1],results['clean_results']['5_points_2_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(results['clean_results']['5_points_2_predators']['POF-based-greedy']['zdepth'][:1],results['clean_results']['5_points_2_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='   POF, POF points, and \nGAMAT Crossover Types')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined/5x2_combined_losses.pdf", bbox_inches='tight')
plt.close(14)




POF = [[0.7 , 0.79365079],
        [2.5, 0.222222222]]

POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 3]
y_axis_height = 1.4

epochs = range(1,11)
plt.figure(15)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(results['test_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'], results['test_results']['2_points_2_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['2_points_2_predators']['antipodes-greedy']['zdepth'], results['test_results']['2_points_2_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(results['test_results']['2_points_2_predators']['POF-based-non-greedy']['zdepth'],results['test_results']['2_points_2_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(results['test_results']['2_points_2_predators']['POF-based-greedy']['zdepth'],results['test_results']['2_points_2_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(results['clean_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'][:1],results['clean_results']['2_points_2_predators']['antipodes-non-greedy']['normal'][:1],'s', color='red' ,label='clean-antipodes-non-greedy')
plt.plot(results['clean_results']['2_points_2_predators']['antipodes-greedy']['zdepth'][:1],results['clean_results']['2_points_2_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
plt.plot(results['clean_results']['2_points_2_predators']['POF-based-non-greedy']['zdepth'][:1],results['clean_results']['2_points_2_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
plt.plot(results['clean_results']['2_points_2_predators']['POF-based-greedy']['zdepth'][:1],results['clean_results']['2_points_2_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)

# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
# legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='  POF, POF points, and \nPrey-Predator Settings')
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='   POF, POF points, and \nGAMAT Crossover Types')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined/2x2_combined_losses.pdf", bbox_inches='tight')
plt.close(15)






















exit()



# REEVALUATED MODELS




loss_results_reevaluated = {
    'test_results': {
           '2_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.2616946493227457, 1.3009359336390938, 1.358414697892887, 1.3621104121208192, 1.349924572595616, 1.348645596774583, 1.3580545727739628, 1.3636364984758123, 1.3655842253842305, 1.3675902622262228, 1.3617110039769988, 1.3666616238269609, 1.3778274968727349, 1.379274092752909, 1.2920071886986801, 1.2937084233637937, 1.3033167410142643, 1.3660783548945004, 1.3602643961758958, 1.3593281667257093, 1.3567779356671363, 1.311893850011924, 1.3121315353924465, 1.3111463919128339, 1.3018005951163696, 1.3130475597283275, 1.3137297318153776, 1.3072003403889765, 1.3077632586980603, 1.319636107351362],
                                    'normal': [0.28366552606685874, 0.2733275740724249, 0.27925736218998115, 0.27744743716470976, 0.27797441789784383, 0.27384053960288923, 0.283476862157743, 0.2847917015097805, 0.2860010892152786, 0.2834575438007866, 0.2778555986500278, 0.28034087514754424, 0.27144985469346194, 0.27240953181207794, 0.2728431756348954, 0.28169060988524525, 0.2878967086371687, 0.28070311162275136, 0.2788982313932832, 0.2776272747012758, 0.27536684889154334, 0.3046236326092297, 0.3040556379200257, 0.2988589912652969, 0.2859601762184163, 0.2899668701530732, 0.2927565546379876, 0.29461462070646977, 0.29627184858641675, 0.2845960971928134]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.0282177376992925, 1.1520035502836876, 1.1546111931505891, 1.1552511716626355, 1.1567004877267424, 1.1556307022104557, 1.1567399723013652, 1.1599838018417359, 1.1571481866934865, 1.1574973754047118, 1.1588940045268266, 1.1582723454101798, 1.1558893640016772, 1.1556070137269718, 1.1539004347988011, 1.1391519299487478, 1.1458844238949806, 1.1491196884322412, 1.1556175088145069, 1.1560950207956058, 1.1562361246531772, 1.1583936357006583, 1.155068536271754, 1.1461057575707583, 1.1492785801592562, 1.1498063163659007, 1.1501327669497619, 1.1572168281397868, 1.156693409521555, 1.1544798003029577],
                                    'normal': [0.23161617432980194, 0.2239881119930867, 0.22335655096265458, 0.22277600221412697, 0.22352655890983406, 0.22403167821082873, 0.22358541548559346, 0.22335224706180318, 0.22353288437595073, 0.22363724985073521, 0.22349457891331506, 0.22347364921852486, 0.22406079750700095, 0.22438292524863765, 0.22401013594005525, 0.22751480433129773, 0.2271086259293802, 0.2259834659161027, 0.22343488094425693, 0.22475411042417448, 0.22430883429406845, 0.2240382179157021, 0.22411324549274347, 0.22551688286139793, 0.22516091593454793, 0.22416618799733132, 0.22415266187535118, 0.22508492114924894, 0.2242902032339696, 0.22728641248240913]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.1464353976790438, 1.2648727590275795, 1.281784321598171, 1.274480689923788, 1.2721977349409124, 1.2870796649726395, 1.276056769582414, 1.3272160644383775, 1.3100464040471107, 1.3043210275394401, 1.288009628438458, 1.3065273364794623, 1.3013418197631836, 1.2961674389150954, 1.2968775655805451, 1.2967778572102182, 1.3802252704335243, 1.3450315595902118, 1.3440973834893137, 1.3644825731356118, 1.3474944260931507, 1.3191856595658764, 1.3340456469771789, 1.3419542170062508, 1.3521222885122004, 1.3437767344651763, 1.3513647515749194, 1.3345887048957275, 1.3211190503897126, 1.2817636269884012],
                                    'normal': [0.2748452302106877, 0.3074891942063558, 0.3034047512356768, 0.30197076539403384, 0.30561919660912346, 0.31369159424427856, 0.30573768382219924, 0.30534312602170965, 0.3113632101373574, 0.3196517766136484, 0.3118309135904017, 0.31451531157051166, 0.31263548329318924, 0.3123509249736353, 0.3123341618124972, 0.31122510731220243, 0.3171891497275264, 0.3200172271003428, 0.31576375660208084, 0.316822606663114, 0.3139239418752415, 0.31422673709613763, 0.3105825002353216, 0.3143716191200866, 0.31329275265182416, 0.31428244209166656, 0.31285407875002047, 0.3117130754841972, 0.3102603130733844, 0.310310674389613]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.232704691297, 1.270941472176424, 1.15834434327391, 1.1070506066391148, 1.0938900732502495, 1.0419962926009267, 1.0332705773029132, 1.0324919119323652, 1.0261202132578977, 1.026296126350914, 1.0327420248198755, 1.032818493031964, 1.0310903476685593, 1.0315469714784131, 1.0335103844858937, 1.0276212799180415, 1.0299033325972016, 1.0298227281914545, 1.03437688719366, 1.0325907924740585, 1.0321449636184064, 1.03015165537903, 1.0295610693312183, 1.0303720347660104, 1.0279480457305907, 1.0287112147537703, 1.0278445879208673, 1.0272160358035687, 1.0324581006138596, 1.0317212982276052],
                                    'normal': [0.32781863765618235, 0.30492064700913185, 0.2747315744149316, 0.27985593504512435, 0.28215394284307344, 0.28667756604779626, 0.2879291679441314, 0.28826152510249736, 0.28835972003715554, 0.28845765977790677, 0.28827571174533095, 0.28796076006496074, 0.28771515368800804, 0.28752954488562554, 0.2885986759183333, 0.288675203427826, 0.28870908089519776, 0.28856707885093297, 0.2885391863043775, 0.2884505366848916, 0.28844974637031556, 0.2884681321910976, 0.2884652628419326, 0.2886029761476615, 0.2884117491466483, 0.2882002704844032, 0.28837591440407273, 0.2882878408296821, 0.2880981271414413, 0.28786898556443835]
                                },
           }, 
           '2_points_5_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.0391022744866991, 1.1819963871818229, 1.2339900673050241, 1.1464595183883746, 1.0747576700043433, 1.0765771647089535, 1.0935774761376922, 1.1935401937396255, 1.1582199739426682, 1.1301624118667288, 1.1092897423763866, 1.0996703746392555, 1.140426027283226, 1.1796950066212526, 1.1529172881362364, 1.1889712808058435, 1.1163883745055838, 1.1152244456035574, 1.091323833244363, 1.1166307760268142],
                                    'normal': [0.3002208948749857, 0.26027783232251395, 0.2559019701689789, 0.2857443496431272, 0.2847076531230789, 0.2846912906341946, 0.2866273729149828, 0.30749025774985245, 0.3068798098367514, 0.3113892636655532, 0.28501476132378134, 0.286172071590866, 0.3063662909969841, 0.3186797993392059, 0.30565629198993605, 0.31068038276790344, 0.3093779084301486, 0.3184852241548066, 0.32458476239872963, 0.3189204267619811]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.0580638166555425, 1.0161533236503602, 1.03967332127168, 1.0424203711686675, 1.041880299381374, 1.0426895489397736, 1.0425273414739629, 1.043061633577052, 1.0441045045852662, 1.0371855295810504, 1.036895327961322, 1.038877418852344, 1.037862067738759, 1.0417552838620452, 1.035299672416805, 1.1072215319908771, 1.1177294329269645, 1.1179010682499286, 1.1096475463552573, 1.1112119533352016],
                                    'normal': [0.3076504937459513, 0.31165793276939197, 0.3134836209496272, 0.31323251788763656, 0.31301316108900246, 0.313447467260754, 0.31322640417162906, 0.3132582978796713, 0.3134517559685658, 0.31340951735211403, 0.3134480842609995, 0.3130552996065199, 0.3129345749764098, 0.3129680036576753, 0.31386631506005513, 0.3094537935920597, 0.2979980291779508, 0.2988917765850873, 0.2992555661607034, 0.2983210829422646]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.0172425597598873, 1.204439223795822, 1.2213726536514833, 1.2425632822144892, 1.2087206394402021, 1.2469169611783373, 1.2853981461721598, 1.272486734513155, 1.2922989152141453, 1.2791890923509892, 1.2662376199800944, 1.3087471749364716, 1.2886090262648986, 1.2866836909166317, 1.280665992461529, 1.2771365401671104, 1.299808745654588, 1.2964269491815075, 1.2955905223630138, 1.299015143851644],
                                    'normal': [0.2853371960600627, 0.31772626656846903, 0.31148141306085686, 0.31357749848021677, 0.3143349535993694, 0.3140829526886498, 0.3167295438727153, 0.32238394034277534, 0.3254670227311321, 0.3258374768741352, 0.3241673953754386, 0.32167883791874363, 0.32176015527592494, 0.3245611390194942, 0.3262219327319529, 0.32544142588512187, 0.3246164821165124, 0.3199088057599117, 0.324183602585006, 0.31934129598828936]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.002618792130775, 1.15884630237658, 1.1543040936755151, 1.192370827665034, 1.2875070528885753, 1.2597745349726726, 1.30731695300525, 1.3461437385106825, 1.3383466156487613, 1.3541353513285057, 1.3401132205097945, 1.3359823691476251, 1.3349018197698692, 1.379395215290109, 1.3416605956775627, 1.3232496208751325, 1.3506242034361535, 1.3738564051303668, 1.375853524011435, 1.1649589875309738, 1.3594549916454197, 1.1719155638488297],
                                    'normal': [0.2370438838742443, 0.2320167098463196, 0.2325964150969515, 0.2622274299257809, 0.24376220067137294, 0.239574372706954, 0.25180316837792543, 0.2561488305170512, 0.2574882679686104, 0.24548758646262062, 0.25251002655815835, 0.24988942831447444, 0.2531111515981635, 0.26689709833602315, 0.26579449674517835, 0.2624755935877869, 0.2603065485499569, 0.27314333660700885, 0.28772183822602343, 0.24263262358522908, 0.2990248848789746, 0.2392711931897193]
                                },
           }, 
           '5_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.0882548052011076, 1.17872775682469, 1.2480767494624423, 1.2836779872166741, 1.2850160565572915, 1.2757204815284493, 1.2703687009123183, 1.2748149890260598, 1.2808691006345847, 1.2566655682534287, 1.258428075510202, 1.2390587290537727, 1.2670271900511278, 1.2643097213863097, 1.2654046759162982, 1.2700439494909699, 1.275119238293048, 1.238229244640193, 1.2485594202562706, 1.2606003144352707],
                                    'normal': [0.25904737606491013, 0.25981850006531193, 0.2544190806519125, 0.2613092439076335, 0.2598453767213625, 0.2630731205964826, 0.2618826491009329, 0.26210444710918307, 0.2619364674251104, 0.2591801611725817, 0.2606575949904845, 0.2601849924043282, 0.2619448252559937, 0.26221157112686905, 0.26014359694780764, 0.26184537293984717, 0.2610720963822198, 0.2536813433022843, 0.2605664508244426, 0.26112115635085353]
                                },
                                # 'antipodes-greedy':       {
                                #     'zdepth': [0.291769704806436,0.5552906052967936,0.8162850905939476,0.3737242487288013,0.2629501870919749,0.7323224939021867,0.4815673781424454,0.6131735555904427,0.5545596857046343,0.5881839815488795,0.7998202572163847,0.7080178205499944,0.5989813267570181,0.6829460311181766,0.9189245008930718,0.8378677913823078,0.7373249183610543,0.5196583640944098,0.6942566955826946,0.6038163682849137],
                                #     'normal': [0.38473424948367874,0.41404957924921487,0.39794838330180377,0.399599816504213,0.4018961911348952,0.4012919439482935,0.41633003433955085,0.41004853973683625,0.4067585883066826,0.3907255092232498,0.40056473385427416,0.4073986637223627,0.4069035960226944,0.3947083865244364,0.3938833671746795,0.3961176082031014,0.4053258003033314,0.40051821980279745,0.4083482199108478,0.40185111607472923]
                                # },
                                # 'POF-based-non-greedy':   {
                                #     'zdepth': [0.2594902884407142,0.22356506026897233,0.21362884613042024,0.2196369659654873,0.21376796306408558,0.21422868681322668,0.21390777860105653,0.2133841877745599,0.2136008228530589,0.21459955044628418,0.21481184406378834,0.21481806792549252,0.21607564291388717,0.21459725357822537,0.21453317065828853,0.21421746866604716,0.2129007021790927,0.2142419652840526,0.21442488914298027,0.21461723274791364],
                                #     'normal': [0.41067553961399905,0.44569360036210914,0.4359559579608367,0.4441886106102737,0.4361191777838874,0.4364648206332295,0.4359670073715682,0.4353521050866117,0.43574016432172247,0.43634790870332224,0.4353447270147579,0.4351433217525482,0.4381571071049602,0.43586194951509694,0.4351047221532802,0.4350099296299453,0.434605160386292,0.4340599156532091,0.4358825196310417,0.43586223070154484]
                                # },
                                'POF-based-greedy':        {
                                    'zdepth': [1.0065875873123247, 1.202972382491397, 1.2104475001698918, 1.1911959182355822, 1.2127757407955289, 1.2067159268044934, 1.183619610304685, 1.21074469187825, 1.2033706535998079, 1.2020096866125913, 1.2089048082066565, 1.2139689874403254, 1.188365644523778, 1.199588117894438, 1.2217310431077308, 1.2047712844671663, 1.1788621000407897, 1.2109680345377971, 1.2148296007176036, 1.1960826036856347],
                                    'normal': [0.24491182725454114, 0.21901680177327285, 0.22308233469417416, 0.22034867972750025, 0.22280392556153622, 0.22345389785225858, 0.22470237237890972, 0.22290384648079725, 0.22205919270048435, 0.22129364508328978, 0.22258743815815327, 0.21861948499974515, 0.21711289870370296, 0.2179656829723378, 0.20962517879058407, 0.21302092304242026, 0.21954265020250044, 0.2096798817521518, 0.20825716816272932, 0.2064414339888956]
                                },
           }, 
           '3_points_3_predators' : {
                                # 'antipodes-non-greedy' :  {
                                #     'zdepth': [0.7601884601042442,0.8082322651577979,0.5696713909660418,3.495925508086214,4.203494570427334,2.412003655777764,3.524550032861454,3.2374448781160963,2.232770092462756,2.1284509968511838,2.038072397536838,2.37800939255154,2.3773627217282955,1.9671169072082362,2.0570844075114456,2.0101054781490992,1.9931238953600225,1.9832626844189831,1.9617956021397385,1.9741365091087892,2.201992493314841,2.1227352078427972,2.3859235389945432,2.405543951644111,2.4420647070579924],
                                #     'normal': [0.4479741981349041,0.44454800990438953,0.44122795966482653,0.4719152521841305,0.4575625548043202,0.4471175153845364,0.45397498914875933,0.4639890872326094,0.4451281094673983,0.43716502318677214,0.4504552731808928,0.42852493229600574,0.4216870685828101,0.4469508996943838,0.44456220226189525,0.4430040868287234,0.4403325013893167,0.44094521925621427,0.44437890956082293,0.44286463641628776,0.449497335657631,0.4499856060927676,0.4220516819929339,0.42800984106113,0.4271337071644891]
                                # },
                                # 'antipodes-greedy':       {
                                #     'zdepth': [1.8493163221890165,1.333176331667556,1.3906840970835734,1.4192199864338353,1.390094046248603,1.3463230486997624,1.347704731557787,1.3973035079916727,1.4013792755677528,1.376394323958564,1.3470690260228422,1.421734890495379,1.3825306961216877,1.3768504897343743,1.3903028345599617,1.3856223556184277,1.3720077541685596,0.9588021924815227,1.5468104974510744,1.0975051360031993],
                                #     'normal': [0.42905900048226425,0.4317897027300805,0.4384664708191586,0.43996092044201096,0.4381895949545595,0.43498568289058726,0.4350637505963906,0.4386642397679004,0.43914765703309444,0.43692411184310914,0.4356705544535647,0.44107162952423096,0.43835168652927753,0.43782244060457365,0.438511790074024,0.43872030746076524,0.4364437968460555,0.44622098072287963,0.458731790977655,0.4528521795862729]
                                # },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.000393912718468, 1.034054351959032, 1.0312165324220952, 1.0403360172645333, 1.063770123117978, 1.046679308488197, 1.0432846698564353, 1.1133801396360103, 1.0726795508689486, 1.095313455394863, 1.0932901167377984, 1.0952101410049753, 1.0933982369826012, 1.1220260386614456, 1.0955570776437975, 1.1365955468305606, 1.1536822239148248, 1.1002552735436824, 1.0709799748106101, 1.1098605144884168],
                                    'normal': [0.2910714610950234, 0.25221569421365087, 0.25167269095317607, 0.2475727031218637, 0.26160825973319024, 0.26475984217579834, 0.264930417642151, 0.25603106516538204, 0.2537569759740043, 0.2378373137761637, 0.23860022413361934, 0.23855852927129292, 0.237653901251321, 0.23497386390400915, 0.2344847167582856, 0.24192979842731632, 0.23432729858713053, 0.2448042433593691, 0.29516969541298976, 0.28424794870553555]
                                },
                                'POF-based-greedy':       {
                                    'zdepth':  [0.8383020997661905, 1.1501293837409658, 1.1503484945936302, 1.1230175728650438, 1.1480906665939645, 1.1148538137219617, 1.1167290177541909, 1.1194068643235668, 1.119784766251279, 1.1191714832463215, 1.1344799478029468, 1.1484116478064625, 1.1539267412166005, 1.143867754690426, 1.1557209209068535, 1.1569793068256575, 1.1657168796382, 1.1761801507055145, 1.177133584759899, 1.1597992148595986],
                                    'normal': [0.3083139116002112, 0.23310421439482995, 0.23346673618886887, 0.2310271783587859, 0.2308355447096923, 0.23023356183902505, 0.23022774633058568, 0.23264720544372638, 0.23228310735262547, 0.23222928392825667, 0.22655652527956618, 0.22927568049467717, 0.2226840987340691, 0.2289228669454142, 0.2281130871975545, 0.22721111073936384, 0.22040056102361877, 0.21678626042665894, 0.21827237444747355, 0.2227244667939304]
                                },
           }, 
        },
    'clean_results': {
           '2_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916, 0.8540316554688916],
                                    'normal': [0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782, 0.23111426009959782]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
           }, 
           '2_points_5_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
           }, 
           '5_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834, 0.8540317087443834],
                                    'normal': [0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336, 0.2311142641705336]
                                },
                                # 'antipodes-greedy':           {
                                #     'zdepth': [1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872, 1.0451087189703872],
                                #     'normal': [0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154, 0.36262792109214154]
                                # },
                                # 'POF-based-non-greedy':   {
                                #     'zdepth': [0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324, 0.43034114023459324],
                                #     'normal': [0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384, 0.38488835188531384]
                                # },
                                # 'POF-based-greedy':       {
                                #     'zdepth': [0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023, 0.4802078352761023],
                                #     'normal': [0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314, 0.37219396276572314]
                                # },
           }, 
        #    '3_points_3_predators' : {
        #                     'antipodes-non-greedy' :       {
        #                             'zdepth': [1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776, 1.5741111476396776],
        #                             'normal': [0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707, 0.38028367100302707]
        #                         },
        #                         'antipodes-greedy':       {
        #                             'zdepth': [1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704, 1.0113691983763704],
        #                             'normal': [0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797, 0.36377969789750797]
        #                         },
        #                         'POF-based-non-greedy':   {
        #                             'zdepth': [0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177, 0.9152095234271177],
        #                             'normal': [0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044, 0.4006406015342044]
        #                         },
        #                         'POF-based-greedy':       {
        #                             'zdepth': [0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551, 0.5789582216862551],
        #                             'normal': [0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744, 0.37983729974510744]
        #                         },
        #    }, 
        },
    
}

POF = [[0.7 , 0.79365079],
        [1.45  , 0.38314176],
        [2.5, 0.222222222]]

POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 3]
y_axis_height = 1.4

epochs = range(1,11)
plt.figure(12)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

# plt.plot(loss_results_reevaluated['test_results']['3_points_3_predators']['antipodes-non-greedy']['zdepth'], loss_results_reevaluated['test_results']['3_points_3_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
# plt.plot(loss_results_reevaluated['test_results']['3_points_3_predators']['antipodes-greedy']['zdepth'], loss_results_reevaluated['test_results']['3_points_3_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['3_points_3_predators']['POF-based-non-greedy']['zdepth'],loss_results_reevaluated['test_results']['3_points_3_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['3_points_3_predators']['POF-based-greedy']['zdepth'],loss_results_reevaluated['test_results']['3_points_3_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(loss_results_reevaluated['clean_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_2_predators']['antipodes-non-greedy']['normal'][:1],'s', color='k' ,label='clean MT-DNN')
# plt.plot(loss_results_reevaluated['clean_results']['3_points_3_predators']['antipodes-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['3_points_3_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['3_points_3_predators']['POF-based-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['3_points_3_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['3_points_3_predators']['POF-based-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['3_points_3_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)

# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='       POF, POF Points,\nGAMAT Crossover Types\n     And Clean MT-DNN')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined_reevaluated/3X3_combined_losses.pdf", bbox_inches='tight')
plt.close(12)



POF = [[0.7 , 0.79365079],
        [2.5, 0.222222222]]

POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 3]
y_axis_height = 1.4



epochs = range(1,11)
plt.figure(13)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(loss_results_reevaluated['test_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'], loss_results_reevaluated['test_results']['2_points_5_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['2_points_5_predators']['antipodes-greedy']['zdepth'], loss_results_reevaluated['test_results']['2_points_5_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['2_points_5_predators']['POF-based-non-greedy']['zdepth'],loss_results_reevaluated['test_results']['2_points_5_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['2_points_5_predators']['POF-based-greedy']['zdepth'],loss_results_reevaluated['test_results']['2_points_5_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(loss_results_reevaluated['clean_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_5_predators']['antipodes-non-greedy']['normal'][:1],'s', color='k' ,label='clean MT-DNN')
# plt.plot(loss_results_reevaluated['clean_results']['2_points_5_predators']['antipodes-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_5_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['2_points_5_predators']['POF-based-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_5_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['2_points_5_predators']['POF-based-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_5_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='       POF, POF Points,\nGAMAT Crossover Types\n     And Clean MT-DNN')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined_reevaluated/2x5_combined_losses.pdf", bbox_inches='tight')
plt.close(13)


POF = [[0.7   , 0.79365079],
    [1.05  , 0.52910053],
    [1.45  , 0.38314176],
    [1.9   , 0.29239766],
    [2.5   , 0.22222222]]   


POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 3]
y_axis_height = 1.4

epochs = range(1,11)
plt.figure(14)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(loss_results_reevaluated['test_results']['5_points_2_predators']['antipodes-non-greedy']['zdepth'], loss_results_reevaluated['test_results']['5_points_2_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
# plt.plot(loss_results_reevaluated['test_results']['5_points_2_predators']['antipodes-greedy']['zdepth'], loss_results_reevaluated['test_results']['5_points_2_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
# plt.plot(loss_results_reevaluated['test_results']['5_points_2_predators']['POF-based-non-greedy']['zdepth'],loss_results_reevaluated['test_results']['5_points_2_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['5_points_2_predators']['POF-based-greedy']['zdepth'],loss_results_reevaluated['test_results']['5_points_2_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(loss_results_reevaluated['clean_results']['5_points_2_predators']['antipodes-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['5_points_2_predators']['antipodes-non-greedy']['normal'][:1],'s', color='k' ,label='clean MT-DNN')
# plt.plot(loss_results_reevaluated['clean_results']['5_points_2_predators']['antipodes-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['5_points_2_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['5_points_2_predators']['POF-based-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['5_points_2_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['5_points_2_predators']['POF-based-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['5_points_2_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)
# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='       POF, POF Points,\nGAMAT Crossover Types\n     And Clean MT-DNN')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined_reevaluated/5x2_combined_losses.pdf", bbox_inches='tight')
plt.close(14)


POF = [[0.7 , 0.79365079],
        [2.5, 0.222222222]]

POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
x_axis_width = [0.000001, 3]
y_axis_height = 1.4

epochs = range(1,11)
plt.figure(15)
plt.ylim(0,1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)

plt.plot(loss_results_reevaluated['test_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'], loss_results_reevaluated['test_results']['2_points_2_predators']['antipodes-non-greedy']['normal'], color='red' ,label='antipodes-non-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['2_points_2_predators']['antipodes-greedy']['zdepth'], loss_results_reevaluated['test_results']['2_points_2_predators']['antipodes-greedy']['normal'], color='blue' ,label='antipodes-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['2_points_2_predators']['POF-based-non-greedy']['zdepth'],loss_results_reevaluated['test_results']['2_points_2_predators']['POF-based-non-greedy']['normal'], color='green' ,label='POF-based-non-greedy',marker='.',ls = '')
plt.plot(loss_results_reevaluated['test_results']['2_points_2_predators']['POF-based-greedy']['zdepth'],loss_results_reevaluated['test_results']['2_points_2_predators']['POF-based-greedy']['normal'], color='cyan' ,label='POF-based-greedy',marker='.',ls = '')
 
plt.plot(loss_results_reevaluated['clean_results']['2_points_2_predators']['antipodes-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_2_predators']['antipodes-non-greedy']['normal'][:1],'s', color='k' ,label='clean MT-DNN')
# plt.plot(loss_results_reevaluated['clean_results']['2_points_2_predators']['antipodes-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_2_predators']['antipodes-greedy']['normal'][:1],'s', color='blue' ,label='clean-antipodes-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['2_points_2_predators']['POF-based-non-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_2_predators']['POF-based-non-greedy']['normal'][:1],'s', color='green' ,label='clean-POF-based-non-greedy')
# plt.plot(loss_results_reevaluated['clean_results']['2_points_2_predators']['POF-based-greedy']['zdepth'][:1],loss_results_reevaluated['clean_results']['2_points_2_predators']['POF-based-greedy']['normal'][:1],'s', color='cyan' ,label='clean-POF-based-greedy')

# plt.xticks(epochs[::1])
plt.xlabel("Average Z-Depth Loss", fontsize=12)
plt.ylabel("Average Normal Loss", fontsize=12)

# plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " -- Testing Fitness ")
# legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='  POF, POF points, and \nPrey-Predator Settings')
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='       POF, POF Points,\nGAMAT Crossover Types\n     And Clean MT-DNN')
plt.setp(legend.get_title(), fontsize='large', fontweight='bold')

plt.savefig("./plots_2/losses/combined_reevaluated/2x2_combined_losses.pdf", bbox_inches='tight')
plt.close(15)

