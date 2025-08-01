def get_educational_content():
    """Return educational content about space weather"""
    
    content = [
        {
            'title': 'What is Space Weather?',
            'content': """
            Space weather refers to the environmental conditions in space as influenced by solar activity and the 
            solar wind's interaction with Earth's magnetosphere. Unlike terrestrial weather, space weather involves 
            electromagnetic phenomena that can significantly impact technology and human activities.
            
            <br><br><strong>Key Components:</strong>
            <ul>
            <li><strong>Solar Wind:</strong> A stream of charged particles continuously flowing from the Sun</li>
            <li><strong>Magnetic Fields:</strong> Complex magnetic field interactions between the Sun and Earth</li>
            <li><strong>Radiation:</strong> High-energy particles and electromagnetic radiation from solar events</li>
            <li><strong>Plasma Environment:</strong> Ionized gas in space affecting radio wave propagation</li>
            </ul>
            
            <br>Understanding space weather is crucial for protecting our technology-dependent society from 
            potentially catastrophic disruptions.
            """,
            'image_caption': 'Artist impression of space weather interaction with Earth'
        },
        
        {
            'title': 'Solar Flares: Explosive Solar Events',
            'content': """
            Solar flares are intense bursts of electromagnetic radiation from the Sun's surface, classified by their 
            X-ray brightness into categories: A, B, C, M, and X (with X being the most powerful).
            
            <br><br><strong>Flare Classifications:</strong>
            <ul>
            <li><strong>C-Class:</strong> Small flares with minimal Earth impact</li>
            <li><strong>M-Class:</strong> Medium flares causing brief radio blackouts</li>
            <li><strong>X-Class:</strong> Large flares capable of causing planet-wide radio blackouts and radiation storms</li>
            </ul>
            
            <br><strong>Impact on Technology:</strong>
            <ul>
            <li>Satellite damage and operational anomalies</li>
            <li>Radio communication disruptions</li>
            <li>GPS navigation errors</li>
            <li>Power grid fluctuations</li>
            <li>Airline route disruptions over polar regions</li>
            </ul>
            
            <br>Solar flares travel at the speed of light, reaching Earth in approximately 8 minutes, making 
            real-time monitoring and rapid response critical for infrastructure protection.
            """,
            'image_caption': 'Solar flare eruption captured by solar observation satellites'
        },
        
        {
            'title': 'Coronal Mass Ejections (CMEs)',
            'content': """
            Coronal Mass Ejections are massive bursts of solar plasma and magnetic field released from the Sun's 
            corona into space. Unlike solar flares, CMEs involve physical matter traveling through space.
            
            <br><br><strong>CME Characteristics:</strong>
            <ul>
            <li><strong>Speed:</strong> Travel at 300-3000 km/s (1-4 days to reach Earth)</li>
            <li><strong>Composition:</strong> Billions of tons of plasma and magnetic field</li>
            <li><strong>Size:</strong> Can be larger than Earth itself</li>
            <li><strong>Direction:</strong> Earth-directed CMEs pose the greatest threat</li>
            </ul>
            
            <br><strong>Geomagnetic Storm Effects:</strong>
            <ul>
            <li>Power grid overloads and blackouts</li>
            <li>Satellite orbit decay and system failures</li>
            <li>Enhanced auroral activity</li>
            <li>Pipeline corrosion acceleration</li>
            <li>Compass navigation errors</li>
            </ul>
            
            <br>The 1859 Carrington Event was the most powerful geomagnetic storm in recorded history, caused by 
            a massive CME. A similar event today could cause trillions of dollars in damage to modern infrastructure.
            """,
            'image_caption': 'CME propagation from Sun toward Earth'
        },
        
        {
            'title': 'Geomagnetic Storms and Earth\'s Response',
            'content': """
            Geomagnetic storms occur when solar wind disturbances interact with Earth's magnetosphere, causing 
            significant variations in the geomagnetic field. These storms are measured using the Kp index (0-9 scale).
            
            <br><br><strong>Storm Categories:</strong>
            <ul>
            <li><strong>G1 (Minor):</strong> Kp=5 - Weak power grid fluctuations</li>
            <li><strong>G2 (Moderate):</strong> Kp=6 - Spacecraft charging, aurora at high latitudes</li>
            <li><strong>G3 (Strong):</strong> Kp=7 - Satellite navigation degraded, power systems affected</li>
            <li><strong>G4 (Severe):</strong> Kp=8 - Widespread charging problems, power grid instability</li>
            <li><strong>G5 (Extreme):</strong> Kp=9 - Power grid collapse possible, satellite control problems</li>
            </ul>
            
            <br><strong>Economic Impact:</strong>
            <br>Geomagnetic storms cause an estimated $10-70 billion annually in economic losses through:
            <ul>
            <li>Power outage costs and infrastructure replacement</li>
            <li>Satellite mission failures and reduced operational life</li>
            <li>Aviation rerouting and communication delays</li>
            <li>Oil pipeline monitoring system disruptions</li>
            </ul>
            
            <br>Modern society's increasing dependence on technology makes geomagnetic storm monitoring and 
            prediction essential for economic stability and public safety.
            """,
            'image_caption': 'Global geomagnetic field disturbance visualization'
        },
        
        {
            'title': 'Satellite Vulnerabilities and Space Weather',
            'content': """
            Satellites operate in a harsh space environment where they are constantly exposed to charged particles, 
            radiation, and electromagnetic disturbances from space weather events.
            
            <br><br><strong>Primary Threats:</strong>
            <ul>
            <li><strong>Surface Charging:</strong> Accumulation of electric charge causing component failures</li>
            <li><strong>Deep Dielectric Charging:</strong> Internal charging leading to electronic anomalies</li>
            <li><strong>Single Event Effects:</strong> High-energy particles causing memory errors or latchup</li>
            <li><strong>Total Ionizing Dose:</strong> Cumulative radiation damage over satellite lifetime</li>
            <li><strong>Atmospheric Drag:</strong> Increased atmospheric density during storms affecting orbits</li>
            </ul>
            
            <br><strong>Operational Impacts:</strong>
            <ul>
            <li>GPS accuracy degradation affecting navigation systems</li>
            <li>Communication satellite outages</li>
            <li>Earth observation mission interruptions</li>
            <li>International Space Station crew safety concerns</li>
            <li>Spacecraft orbit decay and premature re-entry</li>
            </ul>
            
            <br><strong>Protection Strategies:</strong>
            <ul>
            <li>Radiation-hardened electronics design</li>
            <li>Autonomous safe mode activation during storms</li>
            <li>Redundant systems and error correction codes</li>
            <li>Strategic orbit adjustments and mission planning</li>
            </ul>
            """,
            'image_caption': 'Satellite in space weather environment showing particle interactions'
        },
        
        {
            'title': 'Power Grid Impacts and Protection',
            'content': """
            Power grids are particularly vulnerable to space weather due to their extensive conductive networks 
            that can act as antennas for geomagnetic disturbances, inducing dangerous currents.
            
            <br><br><strong>Geomagnetically Induced Currents (GICs):</strong>
            <ul>
            <li>Caused by rapid changes in Earth's magnetic field</li>
            <li>Flow through power lines and transformer windings</li>
            <li>Can cause transformer saturation and overheating</li>
            <li>Lead to voltage instability and cascading failures</li>
            </ul>
            
            <br><strong>Historical Events:</strong>
            <ul>
            <li><strong>1989 Quebec Blackout:</strong> 6 million people lost power for 9 hours</li>
            <li><strong>2003 Swedish Power Outage:</strong> Transformer damage in Malmö</li>
            <li><strong>1989 New Jersey Event:</strong> $6 million transformer destroyed</li>
            </ul>
            
            <br><strong>Protection Measures:</strong>
            <ul>
            <li>Real-time GIC monitoring systems</li>
            <li>Transformer neutral current blocking devices</li>
            <li>Operational procedures for space weather events</li>
            <li>Strategic reserve management and load shedding</li>
            <li>Improved space weather forecasting integration</li>
            </ul>
            
            <br>A Carrington-level event today could cause $1-2 trillion in damage to the North American power grid 
            alone, with recovery taking 4-10 years due to transformer replacement challenges.
            """,
            'image_caption': 'Power grid network showing GIC flow patterns during geomagnetic storms'
        },
        
        {
            'title': 'Space Weather Monitoring and Prediction',
            'content': """
            Effective space weather monitoring requires a global network of ground-based and space-based instruments 
            providing real-time data on solar activity and its effects on Earth's environment.
            
            <br><br><strong>Monitoring Infrastructure:</strong>
            <ul>
            <li><strong>Solar Observation:</strong> SOHO, SDO, Parker Solar Probe missions</li>
            <li><strong>Magnetosphere Monitoring:</strong> ACE, DSCOVR, THEMIS satellites</li>
            <li><strong>Ground Networks:</strong> Magnetometer chains, riometers, GPS networks</li>
            <li><strong>Ionospheric Monitoring:</strong> Digisonde networks, incoherent scatter radars</li>
            </ul>
            
            <br><strong>Prediction Challenges:</strong>
            <ul>
            <li>Complex nonlinear dynamics of the Sun-Earth system</li>
            <li>Limited lead time for accurate forecasting</li>
            <li>Regional variations in space weather effects</li>
            <li>Coupling between different space weather phenomena</li>
            </ul>
            
            <br><strong>Forecast Products:</strong>
            <ul>
            <li>27-day solar activity outlook</li>
            <li>3-day geomagnetic activity forecast</li>
            <li>Real-time alerts and warnings</li>
            <li>Probabilistic impact assessments</li>
            </ul>
            
            <br>Machine learning and artificial intelligence are revolutionizing space weather prediction, enabling 
            more accurate forecasts and better understanding of complex space weather processes.
            """,
            'image_caption': 'Global space weather monitoring network illustration'
        },
        
        {
            'title': 'Solar Cycle and Long-term Variability',
            'content': """
            The Sun follows an approximately 11-year cycle of activity, characterized by the number and intensity 
            of sunspots, solar flares, and CMEs. Understanding this cycle is crucial for long-term space weather prediction.
            
            <br><br><strong>Solar Cycle Phases:</strong>
            <ul>
            <li><strong>Solar Minimum:</strong> Few sunspots, lower flare activity, quiet conditions</li>
            <li><strong>Rising Phase:</strong> Increasing activity, more frequent events</li>
            <li><strong>Solar Maximum:</strong> Peak activity, highest risk period</li>
            <li><strong>Declining Phase:</strong> Decreasing activity, but still significant events possible</li>
            </ul>
            
            <br><strong>Current Solar Cycle 25:</strong>
            <ul>
            <li>Began in December 2019</li>
            <li>Expected to peak around 2024-2025</li>
            <li>Predicted to be moderate in intensity</li>
            <li>Increased activity expected through 2024</li>
            </ul>
            
            <br><strong>Long-term Trends:</strong>
            <ul>
            <li>Grand solar minima (like Maunder Minimum 1645-1715)</li>
            <li>Secular variations in magnetic field strength</li>
            <li>Correlation with climate patterns</li>
            <li>Impact on satellite orbital decay rates</li>
            </ul>
            
            <br>While solar maximum periods have higher average activity, extreme events can occur at any phase 
            of the solar cycle, making continuous monitoring essential.
            """,
            'image_caption': 'Solar cycle sunspot number variation over multiple cycles'
        }
    ]
    
    return content

def get_space_weather_glossary():
    """Return glossary of space weather terms"""
    
    glossary = {
        'Aurora': 'Natural light displays in polar regions caused by charged particles interacting with Earth\'s atmosphere',
        'CME': 'Coronal Mass Ejection - massive burst of solar plasma and magnetic field into space',
        'Dst Index': 'Disturbance storm time index measuring geomagnetic storm intensity',
        'F10.7': 'Solar radio flux at 10.7 cm wavelength, indicator of solar activity level',
        'GIC': 'Geomagnetically Induced Currents - currents induced in conductors by magnetic field changes',
        'Heliosphere': 'Region of space dominated by solar wind extending beyond Pluto',
        'IMF': 'Interplanetary Magnetic Field - magnetic field carried by solar wind',
        'Kp Index': 'Planetary K-index measuring global geomagnetic activity (0-9 scale)',
        'L-shell': 'Magnetic shell parameter describing particle drift paths in magnetosphere',
        'Magnetopause': 'Boundary between Earth\'s magnetosphere and solar wind',
        'Parker Spiral': 'Spiral structure of interplanetary magnetic field due to solar rotation',
        'Proton Event': 'Solar energetic particle event with enhanced proton flux',
        'Reconnection': 'Process where magnetic field lines break and reconnect, releasing energy',
        'SEP': 'Solar Energetic Particles - high-energy particles accelerated by solar events',
        'Solar Wind': 'Stream of charged particles continuously flowing from the Sun',
        'Substorm': 'Brief disturbance in Earth\'s magnetosphere causing aurora and magnetic variations',
        'TEC': 'Total Electron Content - measure of ionospheric electron density affecting GPS',
        'Van Allen Belts': 'Radiation belts of trapped particles around Earth'
    }
    
    return glossary

def get_educational_videos():
    """Return list of educational video URLs and descriptions"""
    
    videos = [
        {
            'title': 'Introduction to Space Weather',
            'description': 'Basic overview of space weather phenomena and their effects on Earth',
            'url': 'https://www.youtube.com/watch?v=oHHSSJDJ4oo',  # NASA video
            'duration': '5:30'
        },
        {
            'title': 'Solar Flares and CMEs Explained',
            'description': 'Detailed explanation of solar eruptive events and their propagation',
            'url': 'https://www.youtube.com/watch?v=HFT7ATLQQx8',  # ESA video
            'duration': '8:15'
        },
        {
            'title': 'Geomagnetic Storms and Power Grids',
            'description': 'How space weather affects electrical power systems',
            'url': 'https://www.youtube.com/watch?v=7ukQhycKOFw',  # Educational video
            'duration': '6:45'
        }
    ]
    
    return videos

def get_case_studies():
    """Return historical case studies of significant space weather events"""
    
    case_studies = [
        {
            'title': 'The Carrington Event (1859)',
            'description': """
            The most powerful geomagnetic storm in recorded history occurred on September 1-2, 1859. 
            Telegraph systems worldwide failed, with some operators receiving electric shocks. Aurora 
            were visible as far south as the Caribbean. If a similar event occurred today, it could 
            cause $2-3 trillion in damage and take 4-10 years for complete recovery.
            """,
            'lessons': [
                'Extreme events can exceed all previous observations',
                'Telegraph technology was primitive compared to modern electronics',
                'Global infrastructure vulnerability has increased dramatically',
                'Economic impact would be unprecedented in modern society'
            ]
        },
        {
            'title': 'Quebec Blackout (March 13, 1989)',
            'description': """
            A geomagnetic storm caused the collapse of Quebec's power grid, leaving 6 million people 
            without electricity for 9 hours. The event demonstrated the vulnerability of power grids 
            to space weather and led to improved monitoring and protection measures.
            """,
            'lessons': [
                'Cascading failures can propagate rapidly through power networks',
                'Transformer damage can take months to repair',
                'Economic losses extend far beyond the power sector',
                'Space weather monitoring became critical for grid operators'
            ]
        },
        {
            'title': 'Halloween Storms (October 2003)',
            'description': """
            A series of powerful solar flares and CMEs occurred in late October 2003, causing 
            widespread satellite anomalies, airline diversions, and power outages. The events 
            highlighted the global nature of space weather impacts on modern technology.
            """,
            'lessons': [
                'Multiple events can compound impacts',
                'Satellite constellation vulnerabilities became apparent',
                'Aviation industry developed new protocols',
                'International cooperation in monitoring improved'
            ]
        }
    ]
    
    return case_studies
