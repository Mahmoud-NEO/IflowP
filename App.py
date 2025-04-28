import streamlit as st

def main():
    about = st.Page(
        page='directory/Home.py',
        title='About Project',
        icon='❓',
        default=True
    )

    prediction = st.Page(
        page='directory/Second_Back_Pressure.py',
        title='Back Pressure',
        icon='💡'
    )

    pg = st.navigation(
        {
            'Info': [about],
            'Projects': [prediction]
        }
    )

    pg.run()

if __name__ == '__main__':
    main()


    
    