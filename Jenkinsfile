pipeline {
    agent any

    environment {
        IMAGE_NAME = 'zenonix/mlops-lab'
        IMAGE_TAG  = 'latest'
    }

    stages {

        stage('Source Code Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Dependency Installation') {
            steps {
                dir('2022bcs0179-mlops') {
                    sh '''
                        python3 -m venv .venv
                        .venv/bin/pip install --upgrade pip --quiet
                        .venv/bin/pip install -r requirements.txt --quiet
                    '''
                }
            }
        }

        stage('Model Training') {
            steps {
                dir('2022bcs0179-mlops') {
                    sh '.venv/bin/python src/train.py'
                }
            }
        }

        stage('Metrics Generation') {
            steps {
                dir('2022bcs0179-mlops') {
                    sh 'cat metrics.json'
                }
                archiveArtifacts artifacts: '2022bcs0179-mlops/metrics.json', fingerprint: true
            }
        }

        stage('Docker Image Build') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} 2022bcs0179-mlops/"
            }
        }

        stage('Docker Image Push') {
            when {
                expression { return env.DOCKERHUB_USER?.trim() }
            }
            steps {
                sh """
                    echo \$DOCKERHUB_PASS | docker login -u \$DOCKERHUB_USER --password-stdin
                    docker push ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

    }

    post {
        always {
            cleanWs()
        }
    }
}
